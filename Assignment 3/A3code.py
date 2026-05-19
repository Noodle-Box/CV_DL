################################################# Import Libraries ################################################
# Standard imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Sklearn for performance evaluation
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

# Pytorch and related imports for NN stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data

# MedMNIST dataset 
import medmnist
from medmnist import INFO

# Set up for GPU/CPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


################################# The ResNet-18 Architecture + Residual Block Structure from draft ##################################

# Code to build Residual Block-I and Residual Block-II. From given code draft (untouched)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# Building the ResNet-18 architecture. 
class ResNet18(nn.Module):

    def __init__(self, input_channels=3, channel_nums=None, n_classes=8):
        super(ResNet18, self).__init__()

        # Use hyperparameters defined in main function. Used for tuning and optimizing
        if channel_nums is None:
            channel_nums = [16, 32, 64, 128]

        if len(channel_nums) != 4:
            raise ValueError("channel_nums must contain four values: [C1, C2, C3, C4]")
        self.channel_nums = channel_nums
        c1, c2, c3, c4 = self.channel_nums

        # Initial layer:
        # Conv2d 7x7 stride 2 padding 3 -> BatchNorm2d -> ReLU -> MaxPool2d 7x7 stride 2 padding 1.
        # Input tensor shape from the BloodMNIST dataloader: [batch_size, 3, 28, 28].
        # The 7x7 convolution increases the feature depth from 3 RGB channels to C1
        # feature maps while downsampling the spatial size from 28x28 to 14x14.
        # ReLU introduces non-linearity so the network can learn more complex image
        # patterns than a purely linear stack of convolutions.
        self.relu = nn.ReLU(inplace=True)

        # Group the full initial feature extractor so the forward pass stays compact.
        # BatchNorm2d stabilizes the C1 feature maps before ReLU, and MaxPool2d follows
        # the task sheet exactly. With a 28x28 input image, the output shape becomes
        # [batch_size, C1, 5, 5].
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=c1,
                kernel_size=(7, 7),
                stride=2,
                padding=3,
                dilation=1,
                bias=False
            ),
            nn.BatchNorm2d(c1),
            self.relu,
            nn.MaxPool2d(
                kernel_size=(7, 7),
                stride=2,
                padding=1,
                dilation=1
            )
        )

        # Residual block cascade:
        # Group 1: two Residual Block-I modules using C1.
        # Group 2: one Residual Block-II followed by one Residual Block-I using C2.
        # Group 3: one Residual Block-II followed by one Residual Block-I using C3.
        # Group 4: one Residual Block-II followed by one Residual Block-I using C4.
        # Block-II modules use stride 2 to downsample spatial size and change channel
        # count; Block-I modules use stride 1 to refine features at the same channel count.
        self.block_cascade = nn.Sequential(

            # Group 1
            ResidualBlock(in_channels=c1, out_channels=c1, stride=1),
            ResidualBlock(in_channels=c1, out_channels=c1, stride=1),
            # Group 2
            ResidualBlock(in_channels=c1, out_channels=c2, stride=2),
            ResidualBlock(in_channels=c2, out_channels=c2, stride=1),
            # Group 3
            ResidualBlock(in_channels=c2, out_channels=c3, stride=2),
            ResidualBlock(in_channels=c3, out_channels=c3, stride=1),
            # Group 4
            ResidualBlock(in_channels=c3, out_channels=c4, stride=2),
            ResidualBlock(in_channels=c4, out_channels=c4, stride=1)
        )

        # 2D global average pooling:
        # AdaptiveAvgPool2d((1, 1)) averages each C4 feature map into a single value,
        # so the output shape becomes [batch_size, C4, 1, 1].
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten converts pooled feature maps from [batch_size, C4, 1, 1] to
        # [batch_size, C4], which is the required input shape for fully connected layers.
        self.flatten = nn.Flatten()

        # Fully connected classifier:
        # The first linear layer reduces the C4 feature vector to C4/2 features.
        # The ReLU adds non-linearity before the final linear layer maps features to
        # the 8 BloodMNIST class logits. Softmax is not used here because nn.CrossEntropyLoss expects raw logits.
        self.fully_connected = nn.Sequential(
            nn.Linear(c4, c4 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c4 // 2, n_classes)
        )

    def forward(self, x):
        # Pass the BloodMNIST image batch through the initial feature extractor.
        # Then pass the extracted features through the residual blocks to learn 
        # Higher-level features such as edges, shapes, and textures relevant to blood cell classification.
        # Final output is the raw class logits
        out = self.initial(x)
        out = self.block_cascade(out)
        out = self.global_avg_pool(out)
        out = self.flatten(out)
        out = self.fully_connected(out)

        # Finally return the tensor of raw class logits with shape [128, 8] for batch size and classes
        return out


############################################# Data Loading and Preprocessing for the NN ##############################################

# Main function to load the datasets for model training, validation (parameter tuning) and testing
# This function returns the metadata and the dataloader structures. Used in main 
def load_bloodmnist_data(batch_size, download, size):

    # We utilize the "BloodMNIST" dataset in the "MedMNIST2D" category.
    #
    # Where:
    #
    # Total images = 17,092 images in bloodmnist dataset
    # Each image is a 28x28 pixel with 3 RGB channels, type: float32 after ToTensor()
    # Each label is stored as an integer in range [0, 7] for the 8 classes.
    # Every dataset has structure: (image, label)
    #
    # Training Set = 11959 images
    # Validation Set = 1712 images
    # Testing Set = 3421 images
    #
    # There are 8 different classes (types) of blood cells:
    # 0: Basophils
    # 1: Eosinophils
    # 2: Erythroblasts
    # 3: Immature Granulocytes (IG), which consists of metamyelocytes, myelocytes, and promyelocytes
    # 4: Lymphocytes
    # 5: Monocytes
    # 6: Neutrophils
    # 7: Platelets (thrombocytes)

    # Obtain 2D BloodMNIST dataset and the information of this dataset
    data_type = 'bloodmnist'
    info = INFO[data_type]

    n_classes = len(info['label'])         # Extract the number of classes in this dataset
    input_channel = info['n_channels']     # Extract the number of channels in each image sample (BloodMNIST has images with 3 color channels)

    DataClass = getattr(medmnist, info['python_class'])

    ######### Data Loading and Preprocessing ##########

    # 1. Transform data for PyTorch formatting.
    # Transforms from (number_of_images, height, width, channels) ---> (channels, height, width)
    # In this case: (number_of_images, 28, 28, 3) ---> (3, 28, 28)
    #
    # 2. Normalize the data.
    # Set the mean and standard deviation to 0.5 to map pixel values from [0, 1] to [-1, 1]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = DataClass(split='train', transform=data_transform, download=download, size=size)
    validate_data = DataClass(split='val', transform=data_transform, download=download, size=size)
    test_data = DataClass(split='test', transform=data_transform, download=download, size=size)

    # Confirm that all official BloodMNIST splits were loaded.
    print("BloodMNIST dataset loaded successfully.")

    # Put the training and testing datasets in dataloader structures. 
    # You will use the dataloaders in your training and testing functions to visit each sample of the batch.
    pin_memory = device.type == "cuda"
    train_dataset = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    validate_dataset = data.DataLoader(dataset=validate_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_dataset = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

    return train_dataset, validate_dataset, test_dataset, n_classes, input_channel, info


# Helper function to get the class names from dataset information. thought i'd write this rather than just hardcoding the class names in
def get_class_names(info, n_classes):
    # Convert the MedMNIST label dictionary into an ordered class-name list.
    # This is used for metric tables and confusion matrix legends.
    return [info["label"][str(i)] for i in range(n_classes)]


################################################## RNN Training Functionality ########################################################

# Main training functionality that uses the training dataset and validation dataset for evaluation. 
# Uses back propagation to update model weights
# Returns training history for metrics. Used in Main()
def train(model, train_dataset, validate_dataset, optimizer, criterion, clip, epoch_num):
    # Train the model and validate it after each epoch.
    # This returns the loss and accuracy history needed for report tables and curves.
    history = {
        "train_loss": [],
        "validate_loss": [],
        "train_accuracy": [],
        "validate_accuracy": [],
        "training_time_seconds": 0.0
    }

    start_time = time.time()

    for epoch in range(epoch_num):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataset:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad()       # Initialize the neural network gradient

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping limits large gradients and helps stabilize training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        validate_metrics = evaluate(model, validate_dataset, criterion)
        validate_loss = validate_metrics["loss"]
        validate_accuracy = validate_metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["validate_loss"].append(validate_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validate_accuracy"].append(validate_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epoch_num}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validate_loss:.4f} | Val Acc: {validate_accuracy:.2f}%"
        )

    history["training_time_seconds"] = time.time() - start_time
    return history

################################################## RNN Evaluation Functionality ######################################################

# Evaluates model without updating weights
# Produces output metrics used for evaluation figures and tables.
# Used in both model validation during training and final test evaluation.
def evaluate(model, eval_dataset, criterion, class_names=None, show_report=False, dataset_name="Evaluation"):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in eval_dataset:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.detach().cpu().numpy())
            all_predictions.extend(predicted.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    labels_for_report = list(range(len(class_names))) if class_names is not None else None

    # Classification metrics show class-level performance beyond overall accuracy.
    # zero_division=0 avoids warnings when an untrained model predicts no samples for a class.
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=labels_for_report)
    per_class_precision = metrics.precision_score(
        all_labels,
        all_predictions,
        labels=labels_for_report,
        average=None,
        zero_division=0
    )
    per_class_recall = metrics.recall_score(
        all_labels,
        all_predictions,
        labels=labels_for_report,
        average=None,
        zero_division=0
    )
    macro_precision = metrics.precision_score(
        all_labels,
        all_predictions,
        labels=labels_for_report,
        average="macro",
        zero_division=0
    )
    weighted_precision = metrics.precision_score(
        all_labels,
        all_predictions,
        labels=labels_for_report,
        average="weighted",
        zero_division=0
    )
    macro_recall = metrics.recall_score(
        all_labels,
        all_predictions,
        labels=labels_for_report,
        average="macro",
        zero_division=0
    )
    weighted_recall = metrics.recall_score(
        all_labels,
        all_predictions,
        labels=labels_for_report,
        average="weighted",
        zero_division=0
    )

    class_metrics = []
    if class_names is not None:
        for class_name, precision, recall in zip(class_names, per_class_precision, per_class_recall):
            class_metrics.append({
                "class": class_name,
                "precision": precision,
                "recall": recall
            })

    if show_report:
        print(f"\n{dataset_name} Metrics")
        print(f"Loss: {running_loss / total:.4f}")
        print(f"Accuracy: {100.0 * correct / total:.2f}%")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Weighted Precision: {weighted_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Weighted Recall: {weighted_recall:.4f}")
        print("\nClassification Metrics by Class")
        print("Class | Precision | Recall")
        print("--------------------------")
        for row in class_metrics:
            print(f"{row['class']} | {row['precision']:.4f} | {row['recall']:.4f}")

    return {
        "loss": running_loss / total,
        "accuracy": 100.0 * correct / total,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "class_metrics": class_metrics,
        "confusion_matrix": conf_matrix,
        "labels": all_labels,
        "predictions": all_predictions
    }

########################################## Figure and Table Generation of Evaluation Metrics #########################################


def plot_training_losses(history, hyperparameter_text=None, save_path=None):
    # Plot training and validation loss per epoch.
    # This graph is needed for the report discussion of underfitting and overfitting.
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Training Loss")
    ax.plot(epochs, history["validate_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = "Training and Validation Loss per Epoch"
    if hyperparameter_text is not None:
        title = f"Training and Validation Loss - Model: {hyperparameter_text}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix_figure(test_metrics, class_names, title="Confusion Matrix (Test Set)", hyperparameter_text=None, save_path=None):
    # Plot the confusion matrix as a heatmap for class-level error analysis.
    # This helps identify which blood cell types are confused by the model.
    conf_matrix = test_metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontweight="bold")
    fig.colorbar(image, ax=ax)

    tick_marks = np.arange(len(class_names))
    numeric_labels = [str(index) for index in tick_marks]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(numeric_labels)
    ax.set_yticklabels(numeric_labels)

    threshold = conf_matrix.max() / 2.0
    for row in range(conf_matrix.shape[0]):
        for col in range(conf_matrix.shape[1]):
            ax.text(
                col,
                row,
                format(conf_matrix[row, col], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[row, col] > threshold else "black"
            )

    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")

    legend_text = "\n".join([f"{index}: {class_name}" for index, class_name in enumerate(class_names)])
    fig.text(
        0.02,
        0.03,
        legend_text,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.4"}
    )

    if hyperparameter_text is not None:
        fig.text(0.5, 0.96, f"Model: {hyperparameter_text}", ha="center", fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=0.28, right=0.92, top=0.88, bottom=0.24)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_classification_metrics_table(test_metrics, title="Classification Metrics (Test Set)", hyperparameter_text=None, save_path=None):
    # Plot precision and recall as a table figure.
    # Support and F1-score are excluded to keep the table focused for the report draft.
    table_rows = []
    for row in test_metrics["class_metrics"]:
        table_rows.append([
            row["class"],
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}"
        ])

    table_rows.append(["Macro Average", f"{test_metrics['macro_precision']:.4f}", f"{test_metrics['macro_recall']:.4f}"])
    table_rows.append(["Weighted Average", f"{test_metrics['weighted_precision']:.4f}", f"{test_metrics['weighted_recall']:.4f}"])

    fig_height = 0.5 * len(table_rows) + 1.5
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    if hyperparameter_text is not None:
        title = f"Classification Metrics - Model: {hyperparameter_text}"
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=table_rows,
        colLabels=["Class", "Precision", "Recall"],
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_final_summary_table(history, test_metrics, title="Model Summary Metrics", hyperparameter_text=None, save_path=None):
    # Plot final test metrics and best training/validation metrics in one table.
    # This table is intended for quick comparison between trained model variants.
    table_rows = [
        ["Final Evaluation Metrics", "Value (%)"],
        ["Final Test Accuracy (%)", f"{test_metrics['accuracy']:.2f}"],
        ["Final Test Loss", f"{test_metrics['loss']:.4f}"],
        ["Final Test Macro Precision", f"{test_metrics['macro_precision']:.4f}"],
        ["Final Test Weighted Precision", f"{test_metrics['weighted_precision']:.4f}"],
        ["Final Test Macro Recall", f"{test_metrics['macro_recall']:.4f}"],
        ["Final Test Weighted Recall", f"{test_metrics['weighted_recall']:.4f}"],
        ["Total Training Time (seconds)", f"{history['training_time_seconds']:.2f}"],
        ["", ""],
        ["Training Metrics", "Value (%)"],
        ["Lowest Training Loss", f"{min(history['train_loss']):.4f}"],
        ["Highest Training Accuracy (%)", f"{max(history['train_accuracy']):.2f}"],
        ["Lowest Validation Loss", f"{min(history['validate_loss']):.4f}"],
        ["Highest Validation Accuracy (%)", f"{max(history['validate_accuracy']):.2f}"]
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    if hyperparameter_text is not None:
        title = f"Model Summary Metrics - Model: {hyperparameter_text}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    table = ax.table(
        cellText=table_rows,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row in (0, 9):
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8f2e8")
        if row == 8:
            cell.set_facecolor("#ffffff")
        if col == 0:
            cell.set_text_props(ha="left")

    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_accuracy_table(history, test_metrics):
    # Print training and validation accuracy in a table.
    # The final test accuracy is printed separately because it is only evaluated after training.
    print("\nAccuracy Table")
    print("Epoch | Training Accuracy (%) | Validation Accuracy (%)")
    print("-------------------------------------------------------")

    for epoch, (train_acc, validate_acc) in enumerate(
        zip(history["train_accuracy"], history["validate_accuracy"]),
        start=1
    ):
        print(f"{epoch:5d} | {train_acc:21.2f} | {validate_acc:23.2f}")

    print("-------------------------------------------------------")
    print(f"Final Test Accuracy (%): {test_metrics['accuracy']:.2f}")
    print(f"Final Test Loss: {test_metrics['loss']:.4f}")
    print(f"Final Test Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Final Test Weighted Precision: {test_metrics['weighted_precision']:.4f}")
    print(f"Final Test Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"Final Test Weighted Recall: {test_metrics['weighted_recall']:.4f}")
    print(f"Total Training Time (seconds): {history['training_time_seconds']:.2f}")

################################################# Saving and Testing the Trained Model ################################################



def save_model_checkpoint(model, optimizer, history, test_metrics, checkpoint_path, channel_nums, learning_rate, epoch_num, input_channel, n_classes):
    # Save the trained model and experiment settings.
    # The checkpoint lets assessors reload the model and evaluate it without retraining.
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "channel_nums": channel_nums,
        "learning_rate": learning_rate,
        "epoch_num": epoch_num,
        "input_channel": input_channel,
        "n_classes": n_classes
    }, checkpoint_path)

    print(f"Model checkpoint saved to: {checkpoint_path}")


def test_saved_model(checkpoint_path, test_dataset, n_classes, info):
    # Load a saved model checkpoint and run final evaluation on the test dataset.
    # This function is for reproducibility without rerunning the full training process.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    criterion = nn.CrossEntropyLoss()
    class_names = get_class_names(info, n_classes)

    model = ResNet18(
        input_channels=checkpoint["input_channel"],
        channel_nums=checkpoint["channel_nums"],
        n_classes=checkpoint["n_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model,
        test_dataset,
        criterion,
        class_names=class_names,
        show_report=True,
        dataset_name="Loaded Model Test"
    )

    hyperparameter_text = (
        f"Channels: {checkpoint['channel_nums']} | "
        f"Learning rate: {checkpoint['learning_rate']} | "
        f"Epochs: {checkpoint['epoch_num']}"
    )

    plot_confusion_matrix_figure(
        test_metrics,
        class_names,
        title="Confusion Matrix (Loaded Model Test Set)",
        hyperparameter_text=hyperparameter_text
    )
    plot_classification_metrics_table(
        test_metrics,
        title="Classification Metrics (Loaded Model Test Set)",
        hyperparameter_text=hyperparameter_text
    )
    plot_final_summary_table(
        checkpoint["history"],
        test_metrics,
        title="Final Evaluation and Training Summary (Loaded Model)",
        hyperparameter_text=hyperparameter_text
    )
    plot_training_losses(
        checkpoint["history"],
        hyperparameter_text=hyperparameter_text
    )

    return test_metrics

def run_training_experiment(train_dataset, validate_dataset, test_dataset, n_classes, input_channel, info, channel_nums, learning_rate, epoch_num, clip, checkpoint_path):
    # Train a new model, evaluate it on the test set, save figures, and save a checkpoint.
    # Use this mode when training a fresh hyperparameter experiment.
    class_names = get_class_names(info, n_classes)
    hyperparameter_text = f"Channels: {channel_nums} | Learning rate: {learning_rate} | Epochs: {epoch_num}"

    model = ResNet18(input_channels=input_channel, channel_nums=channel_nums, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = train(model, train_dataset, validate_dataset, optimizer, criterion, clip, epoch_num)
    test_metrics = evaluate(model, test_dataset, criterion, class_names=class_names, show_report=True, dataset_name="Final Test")

    plot_training_losses(
        history,
        hyperparameter_text=hyperparameter_text,
        save_path="model_outputs/Training_Validation_Loss.png"
    )
    plot_confusion_matrix_figure(
        test_metrics,
        class_names,
        title="Confusion Matrix (Test Set)",
        hyperparameter_text=hyperparameter_text,
        save_path="model_outputs/Confusion_Matrix_Test_Set.png"
    )
    plot_classification_metrics_table(
        test_metrics,
        title="Classification Metrics (Test Set)",
        hyperparameter_text=hyperparameter_text,
        save_path="model_outputs/Classification_Metrics_Test_Set.png"
    )
    plot_final_summary_table(
        history,
        test_metrics,
        title="Final Evaluation and Training Summary",
        hyperparameter_text=hyperparameter_text,
        save_path="model_outputs/Model_Summary_Metrics.png"
    )
    print_accuracy_table(history, test_metrics)

    save_model_checkpoint(
        model,
        optimizer,
        history,
        test_metrics,
        checkpoint_path=checkpoint_path,
        channel_nums=channel_nums,
        learning_rate=learning_rate,
        epoch_num=epoch_num,
        input_channel=input_channel,
        n_classes=n_classes
    )

    return history, test_metrics


########################################## Main Function with Hyperparameter Settings ##########################################


# This is the main function
if __name__ == '__main__':

    #### MAIN HYPERPARAMETERS FOR TUNING ####
    #
    # Each model has their own combination of hyperparameters.
    # This specific model is model [X].
    channel_nums = [16, 32, 64, 128]
    learning_rate = 1e-3
    epoch_num = 5

    # Change this one line only:
    # "train" trains a new model. "test_saved_model" loads the checkpoint and evaluates once.
    RUN_MODE = "test_model"

    # Settings for data loading and preprocessing.
    BATCH_SIZE = 128
    DOWNLOAD = True
    SIZE = 28

    # Training settings.
    CLIP = 1
    CHECKPOINT_PATH = "trained_models/Trained_Model.pth"

    train_dataset, validate_dataset, test_dataset, n_classes, input_channel, info = load_bloodmnist_data(
        batch_size=BATCH_SIZE,
        download=DOWNLOAD,
        size=SIZE
    )

    if RUN_MODE == "train":
        history, test_metrics = run_training_experiment(
            train_dataset=train_dataset,
            validate_dataset=validate_dataset,
            test_dataset=test_dataset,
            n_classes=n_classes,
            input_channel=input_channel,
            info=info,
            channel_nums=channel_nums,
            learning_rate=learning_rate,
            epoch_num=epoch_num,
            clip=CLIP,
            checkpoint_path=CHECKPOINT_PATH
        )

    elif RUN_MODE == "test_model":
        loaded_test_metrics = test_saved_model(
            checkpoint_path=CHECKPOINT_PATH,
            test_dataset=test_dataset,
            n_classes=n_classes,
            info=info
        )

    else:
        raise ValueError('RUN_MODE must be either "train" or "test_model"')
