################################################# Import Libraries ################################################
# Standard imports
import os
import csv
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

    def __init__(self, input_channels=3, num_channels=None, class_num=8):
        super(ResNet18, self).__init__()

        if len(num_channels) != 4:
            raise ValueError("num_channels must contain four values: [C1, C2, C3, C4]")
        self.num_channels = num_channels
        c1, c2, c3, c4 = self.num_channels

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
        self.initial = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=c1, kernel_size=(7, 7), 
                                               stride=2, padding=3, dilation=1, bias=False), 
                                               nn.BatchNorm2d(c1), self.relu, 
                                               nn.MaxPool2d(kernel_size=(7, 7), stride=2, padding=1, dilation=1))

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
        # the 8 BloodMNIST class logits. 
        self.fully_connected = nn.Sequential(nn.Linear(c4, c4 // 2), nn.ReLU(inplace=True), nn.Linear(c4 // 2, class_num))

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
def LoadDataBloodMNIST(batch_size, download, size):

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

    class_num = len(info['label'])         # Extract the number of classes in this dataset
    input_channel = info['n_channels']     # Extract the number of channels in each image sample (BloodMNIST has images with 3 color channels)

    DataClass = getattr(medmnist, info['python_class'])

    ######### Data Loading and Preprocessing ##########

    # 1. Transform data for PyTorch formatting.
    # Transforms from (number_of_images, height, width, channels) ---> (channels, height, width)
    # In this case: (number_of_images, 28, 28, 3) ---> (3, 28, 28)
    #
    # 2. Normalize the data.
    # Set the mean and standard deviation to 0.5 to map pixel values from [0, 1] to [-1, 1]
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    DataTRAIN = DataClass(split='train', transform=data_transform, download=download, size=size)
    DataVALIDATE = DataClass(split='val', transform=data_transform, download=download, size=size)
    DataTEST = DataClass(split='test', transform=data_transform, download=download, size=size)

    # Confirm that all official BloodMNIST splits were loaded.
    print("BloodMNIST dataset loaded successfully.")

    # Put the training and testing datasets in dataloader structures. 
    # You will use the dataloaders in your training and testing functions to visit each sample of the batch.
    pin_memory = device.type == "cuda"
    DataTRAIN_Set = data.DataLoader(dataset=DataTRAIN, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    DataVAL_Set = data.DataLoader(dataset=DataVALIDATE, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    DataTEST_Set = data.DataLoader(dataset=DataTEST, batch_size=1, shuffle=False, pin_memory=pin_memory)

    return DataTRAIN_Set, DataVAL_Set, DataTEST_Set, class_num, input_channel, info


# Helper function to get the class names from dataset information. thought i'd write this rather than just hardcoding the class names in
def GetClassNames(info, class_num):
    return [info["label"][str(i)] for i in range(class_num)]


################################################## RNN Training Functionality ########################################################

# Main training functionality that uses the training dataset and validation dataset for evaluation. 
# Uses back propagation to update model weights
# Returns training history for metrics. Used in Main()
def train(model, DataTRAIN_Set, DataVAL_Set, optimizer, criterion, clip, epoch_num):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train time (s)": 0.0
    }

    start_time = time.time()

    # Main training loop. Utlizes Pytorch features
    for epoch in range(epoch_num):
        # Put the model in training mode
        model.train()

        run_loss = 0.0
        correct = 0
        total = 0

        # Loop through training batches and update model weights with backpropagation.
        # Utilizes CrossEntropyLoss as the loss function and ADAM as the optimizer for weight updates.
        for images, labels in DataTRAIN_Set:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad()       # Initialize the neural network gradient

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping limits large gradients and helps stabilize training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Track the losses and training accuracy for each epoch
            run_loss += loss.item() * images.size(0)         # Average loss for current batch
            _, predicted = torch.max(outputs, 1)             # Selects class with highest output logic for each image
            total += labels.size(0)                          # Adds total number of images in batch to total count
            correct += (predicted == labels).sum().item()    # Counts predictions mathed with true labels and adds them to total correct count

        # Epoch metrics
        train_loss = run_loss / total
        train_acc = 100.0 * correct / total

        # Evaluate the model on the validation set without weight updating for final performance
        val_metric = evaluate(model, DataVAL_Set, criterion)
        val_loss = val_metric["loss"]
        val_acc = val_metric["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print traning metrics to terminal
        print(
            f"Epoch [{epoch + 1}/{epoch_num}] "
            f"Training Loss: {train_loss:.4f} | Training Acc: {train_acc:.2f}% | "
            f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%"
        )

    history["train time (s)"] = time.time() - start_time
    # Return the training history. Used for plotting and final metrics.
    return history

################################################## RNN Evaluation Functionality ######################################################

# Evaluates model without updating weights
# Produces output metrics used for evaluation figures and tables.
# Used in both model validation during training and final test evaluation.
def evaluate(model, eDataVAL_Set, criterion, class_names=None):

    model.eval()

    run_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    # Evaluate the model on the validation test without updating the weights
    with torch.no_grad():
        for images, labels in eDataVAL_Set:
            # Move to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            # Predict class logits and calculatess loss for batch
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Acccumulate the loss then predict the class with highest logit
            run_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            # Accuracy updated by counting corrected predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # True and predicted labels moved back to CPU for storage
            # Used in confusion matrix and other metrics
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    report_labels = list(range(len(class_names))) if class_names is not None else None

    # Classification metrics show class-level performance beyond overall accuracy.
    # zero_division=0 avoids warnings when an untrained model predicts no samples for a class.
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=report_labels)
    class_precision = metrics.precision_score(all_labels, all_preds, labels=report_labels, average=None, zero_division=0)
    class_recall = metrics.recall_score(all_labels, all_preds, labels=report_labels, average=None, zero_division=0)
    macro_precision = metrics.precision_score(all_labels, all_preds, labels=report_labels, average="macro", zero_division=0)
    weight_precision = metrics.precision_score(all_labels, all_preds, labels=report_labels, average="weighted", zero_division=0)
    macro_recall = metrics.recall_score(all_labels, all_preds, labels=report_labels, average="macro", zero_division=0)
    weight_recall = metrics.recall_score(all_labels, all_preds, labels=report_labels, average="weighted", zero_division=0)

    # Create structured list off precision and recall values for each blood cell class
    # Used in plotting classification metrics 
    class_metric = []
    if class_names is not None:
        for class_name, precision, recall in zip(class_names, class_precision, class_recall):
            class_metric.append({
                "class": class_name,
                "precision": precision,
                "recall": recall
            })

    return {
        "loss": run_loss / total,
        "accuracy": 100.0 * correct / total,
        "macro_precision": macro_precision,
        "weight_precision": weight_precision,
        "macro_recall": macro_recall,
        "weight_recall": weight_recall,
        "class_metric": class_metric,
        "confusion_matrix": conf_matrix,
        "labels": all_labels,
        "predictions": all_preds
    }

########################################## Figure and Table Generation of Evaluation Metrics #########################################


# Plots the training and validation loss curves over epochs
# We want to observe how well the model learnend from training data over time
# Also observes if model is overfitting from divergence of training and validation loss curves
def plt_TrainingLoss(history, param_text=None, save_path=None):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Figure formatting
    fig, ax = plt.subplots(figsize=(8, 5))
    title = f"Training and Validation Loss - Model: {param_text}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.plot(epochs, history["train_loss"], label="Training Loss")
    ax.plot(epochs, history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


# Plots the confusion matrix as a heatmap for class-specific performance ananlysis
# We want to see which blood specific classes were correctly classified
# We also want to observe which classes were confused with each other as a visual aid for discussion of model
def plt_ConfMatrix(test_metrics, class_names, param_text=None, save_path=None):

    conf_matrix = test_metrics["confusion_matrix"]

    # Figure formatting
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    title = f"Confusion Matrix - Model: {param_text}"
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
            ax.text(col, row, format(conf_matrix[row, col], "d"), ha="center", va="center", 
                    color="white" if conf_matrix[row, col] > threshold else "black")

    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_xlabel("Predicted", fontweight="bold")

    legend_text = "\n".join([f"{index}: {class_name}" for index, class_name in enumerate(class_names)])
    fig.text(0.02, 0.03, legend_text, fontsize=9, va="bottom", ha="left", 
             bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.4"})

    fig.subplots_adjust(left=0.28, right=0.92, top=0.88, bottom=0.24)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

# Plots the precision and recall for class specific classification
# Want to see if model performed well across all classes rather than only achieving an overall high accuracy
def plt_ClassMetrics(test_metrics, param_text=None, save_path=None):

    # Figure formatting
    table_rows = []
    for row in test_metrics["class_metric"]:
        table_rows.append([row["class"], f"{row['precision']:.4f}", f"{row['recall']:.4f}"])

    table_rows.append(["Macro Average", f"{test_metrics['macro_precision']:.4f}", f"{test_metrics['macro_recall']:.4f}"])
    table_rows.append(["Weighted Average", f"{test_metrics['weight_precision']:.4f}", f"{test_metrics['weight_recall']:.4f}"])

    fig_height = 0.5 * len(table_rows) + 1.5
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")
    title = f"Classification Metrics - Model: {param_text}"
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(cellText=table_rows, colLabels=["Class", "Precision", "Recall"], loc="center", cellLoc="center", colLoc="center")
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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

# Small helper function for converting training time (s) --> (minutes, seconds) 
def format_train_time(train_time_seconds):
    total_seconds = int(round(train_time_seconds))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes} min {seconds} sec"


# Plots overall model metrics in table
# Used to evaluate model's final test performance and its training behaviour as a summary
# Also used as a quick model reference for its performance
def plt_SummaryFinal(history, test_metrics, param_text=None, save_path=None):

    table_rows = [
        ["Final Evaluation Metrics", "Value (%)"],
        ["Final Test Accuracy (%)", f"{test_metrics['accuracy']:.2f}"],
        ["Final Test Loss", f"{test_metrics['loss']:.4f}"],
        ["Final Test Macro Precision", f"{test_metrics['macro_precision']:.4f}"],
        ["Final Test Weighted Precision", f"{test_metrics['weight_precision']:.4f}"],
        ["Final Test Macro Recall", f"{test_metrics['macro_recall']:.4f}"],
        ["Final Test Weighted Recall", f"{test_metrics['weight_recall']:.4f}"],
        ["Total Training Time (minutes, seconds)", format_train_time(history["train time (s)"])],
        ["", ""],
        ["Training Metrics", "Value (%)"],
        ["Lowest Training Loss", f"{min(history['train_loss']):.4f}"],
        ["Highest Training Accuracy (%)", f"{max(history['train_acc']):.2f}"],
        ["Lowest Validation Loss", f"{min(history['val_loss']):.4f}"],
        ["Highest Validation Accuracy (%)", f"{max(history['val_acc']):.2f}"]
    ]

    # Figure formatting
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    title = f"Model Summary Metrics - Model: {param_text}"
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

    plt.close(fig)

########################################### Saving the Model and Wrapper Functions for Main ################################################

# This function creates the folder paths for each model experiment.
# Each model has its own folder for the trained checkpoint and its output figures.
# Used with ModelSave() to save model and its outputs so it can easily be reloaded later
def GetModelPaths(model_num):
    base_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_folder, "Pre-Trained Models", f"Model {model_num}")
    trained_model_folder = os.path.join(model_folder, f"Trained Model - Model {model_num}")
    model_outputs_folder = os.path.join(model_folder, f"Model Outputs - Model {model_num}")
    checkpoint_path = os.path.join(trained_model_folder, f"Model {model_num}.pth")

    return checkpoint_path, model_outputs_folder

# This function saves the model entity after training such that it can be reloaded later for testing without retraining. 
def ModelSave(model, optimizer, history, test_metrics, checkpoint_path, num_channels, learn_rate, epoch_num, channel_in, class_num):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "num_channels": num_channels,
        "learn_rate": learn_rate,
        "epoch_num": epoch_num,
        "input_channel": channel_in,
        "channel_in": channel_in,
        "class_num": class_num
    }, checkpoint_path)

    print(f"Model checkpoint saved to: {checkpoint_path}")

# This function is a wrapper for running the full training experiment. Used in Main()
def RunTraining(DataTRAIN_Set, DataVAL_Set, DataTEST_Set, class_num, channel_in, info, num_channels, learn_rate, epoch_num, clip, checkpoint_path, output_folder):
    # Train a new model, evaluate it on the test set, save figures, and save a checkpoint.
    # Use this mode when training a fresh hyperparameter experiment.
    class_names = GetClassNames(info, class_num)
    param_text = f"Channels: {num_channels} | Learning rate: {learn_rate} | Epochs: {epoch_num}"

    model = ResNet18(input_channels=channel_in, num_channels=num_channels, class_num=class_num).to(device)



    # The cross entropy loss function used for multi-class clasisfication as it compares model's raw class score (logits)
    # the correct class labels. Used in gradient descent to update model weights in train()
    # From [5] in report

    # Working theory:
    # 1. Outputs 8 logits and compares the predicted logits against the correct testing class labels
    # 2. True labels are integers [0,7] and calculates the loss with backpropagation (in train())
    # 3. Loss observed and computes gradients using loss.backward() (in train())
    criterion = nn.CrossEntropyLoss() #Change this for other testing functions from internet


    # Adaptive Movement Estimation (ADAM) is an optimizer for updating neural network weights in train()
    #
    # Working theory:
    # 1. loss.backward in train() computes the gradients for the trainable parameters 
    # 2. Adam keeps a moving average of the gradients for each parameter and its squared values for update scaling
    # 3. M.A of parameters = direction | M.A of squared values = scaling
    # 4. It then uses them to update model parameters such that:
    # M.A of parameters = direction to update | M.A of squared values = how much to update
    # From [4] in report
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Train the model and evaluate on the test set and validation set. Use the CrossEntropyLoss as a loss function
    # Also use the ADAM optimizer for weight updates
    history = train(model, DataTRAIN_Set, DataVAL_Set, optimizer, criterion, clip, epoch_num)

    # Then, evaluate trained model on the test set for final performance metrics. Also use the CrossEntropyLoss as a standard multi-class loss function
    test_metrics = evaluate(model, DataTEST_Set, criterion, class_names=class_names)

    # Run these plotting functions to generation figures for model evaluation. These are saved in model[x] --> Model Outputs - Model [x] folders
    plt_TrainingLoss(history, param_text=param_text, save_path=os.path.join(output_folder, "Training_Validation_Loss.png"))
    plt_ConfMatrix(test_metrics, class_names, param_text=param_text, save_path=os.path.join(output_folder, "Confusion_Matrix_Test_Set.png"))
    plt_ClassMetrics(test_metrics, param_text=param_text, save_path=os.path.join(output_folder, "Classification_Metrics_Test_Set.png"))
    plt_SummaryFinal(history, test_metrics, param_text=param_text, save_path=os.path.join(output_folder, "Model_Summary_Metrics.png"))
    
    ModelSave(model, optimizer, history, test_metrics, checkpoint_path=checkpoint_path, num_channels=num_channels,
                          learn_rate=learn_rate, epoch_num=epoch_num, channel_in=channel_in,class_num=class_num)
    return 


# This function is a wrapper for loading a saved model checkpoint and evaluating it on the test set. Mostly used for markers. Used in Main()
def RunTesting(checkpoint_path, DataTEST_Set, class_num, info, output_folder):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use CrossEntropyLoss as a standard multi-class loss function
    criterion = nn.CrossEntropyLoss() #Change this for other testing functions from internet

    # Extract class names from dataset
    class_names = GetClassNames(info, class_num)

    model = ResNet18(input_channels=checkpoint["input_channel"], num_channels=checkpoint["num_channels"], 
                     class_num=checkpoint["class_num"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the loaded model on the test set for final peformance metrics
    test_metrics = evaluate(model, DataTEST_Set, criterion, class_names=class_names)

    # Extract hyperparameter settings 
    param_text = (f"Channels: {checkpoint['num_channels']} |" f"Learning rate: {checkpoint['learn_rate']} | "f"Epochs: {checkpoint['epoch_num']}")

    # Run these plotting functions to generation figures for model evaluation. These are saved in model[x] --> Model Outputs - Model [x] folders
    plt_ConfMatrix(test_metrics, class_names, param_text=param_text, save_path=os.path.join(output_folder, "Loaded_Model_Confusion_Matrix_Test_Set.png"))
    plt_ClassMetrics(test_metrics, param_text=param_text, save_path=os.path.join(output_folder, "Loaded_Model_Classification_Metrics_Test_Set.png"))
    plt_SummaryFinal(checkpoint["history"], test_metrics, param_text=param_text, save_path=os.path.join(output_folder, "Loaded_Model_Summary_Metrics.png"))
    plt_TrainingLoss(checkpoint["history"], param_text=param_text, save_path=os.path.join(output_folder, "Loaded_Model_Training_Validation_Loss.png"))

    return


# Used to load each saved model "entity" and compare performance metrics
# Allows to visualise and best hyperparameter settings across different trained models.
def CompareSavedModels(start_model, end_model):
    # Read each saved checkpoint and extract validation metrics used for model selection.
    # Missing model files are kept in the table so the comparison can be rerun as more models finish.
    headers = [
        "Model",
        "Channels",
        "Learning Rate",
        "Epochs",
        "Highest Val Acc (%)",
        "Val Acc Epoch",
        "Lowest Val Loss",
        "Val Loss Epoch",
        "Final Val Acc (%)",
        "Final Val Loss",
        "Smallest Train-Val Gap (%)",
        "Gap Epoch",
        "Final Train-Val Gap (%)",
        "Training Time",
        "Val Loss Stability"
    ]

    # Table Figure Formatting
    table_rows = []
    base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pre-Trained Models")

    for model_num in range(start_model, end_model + 1):
        checkpoint_path, _ = GetModelPaths(model_num)

        if not os.path.exists(checkpoint_path):
            table_rows.append([model_num, "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
            continue

        # Load model entity and extract the metric history 
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = checkpoint["history"]
        learn_rate_text = f"{checkpoint['learn_rate']:.0e}".replace("e-0", "e-").replace("e+0", "e+")

        highest_val_acc = max(history["val_acc"])
        highest_val_acc_epoch = history["val_acc"].index(highest_val_acc) + 1

        lowest_val_loss = min(history["val_loss"])
        lowest_val_loss_epoch = history["val_loss"].index(lowest_val_loss) + 1

        train_val_gaps = [abs(train_acc - val_acc) for train_acc, val_acc in zip(history["train_acc"], history["val_acc"])]
        smallest_gap = min(train_val_gaps)
        smallest_gap_epoch = train_val_gaps.index(smallest_gap) + 1
        final_gap = abs(history["train_acc"][-1] - history["val_acc"][-1])

        last_val_losses = history["val_loss"][-10:]
        val_loss_stability = max(last_val_losses) - min(last_val_losses)

        # Table formatting
        table_rows.append([
            model_num,
            str(checkpoint["num_channels"]),
            learn_rate_text,
            checkpoint["epoch_num"],
            f"{highest_val_acc:.2f}",
            highest_val_acc_epoch,
            f"{lowest_val_loss:.4f}",
            lowest_val_loss_epoch,
            f"{history['val_acc'][-1]:.2f}",
            f"{history['val_loss'][-1]:.4f}",
            f"{smallest_gap:.2f}",
            smallest_gap_epoch,
            f"{final_gap:.2f}",
            format_train_time(history["train time (s)"]),
            f"{val_loss_stability:.4f}"
        ])

    os.makedirs(base_folder, exist_ok=True)

    csv_path = os.path.join(base_folder, f"Model Comparison Tables {start_model} to {end_model}.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(table_rows)

    # Output comparison table to console as CSV and load as a figure
    print("\nModel Comparison Table")
    column_widths = [max(len(str(row[index])) for row in [headers] + table_rows) for index in range(len(headers))]
    print(" | ".join(str(headers[index]).ljust(column_widths[index]) for index in range(len(headers))))
    print("-+-".join("-" * width for width in column_widths))
    for row in table_rows:
        print(" | ".join(str(row[index]).ljust(column_widths[index]) for index in range(len(headers))))

    fig_height = 0.45 * len(table_rows) + 2.0
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis("off")
    ax.set_title(f"Model Comparison Table - Models {start_model} to {end_model}", fontsize=12, fontweight="bold", pad=12)

    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")

    fig.tight_layout()

    figure_path = os.path.join(base_folder, f"Model Comparison Figure{start_model} to {end_model}.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nModel comparison CSV saved to: {csv_path}")
    print(f"Model comparison figure saved to: {figure_path}")

    return table_rows

########################################## Main Function with Hyperparameter Settings ##########################################

# This is the main function
if __name__ == '__main__':

    # FOR MARKER: 
    # Change this to "test_model" to run individual model testing
    # Change this to "compare_models" and change (start_model=x, end_model=y) in CompareSavedModels() for overall model commparsion between ranges x and y. 
    # For all, (x,y) use (1, 13). 
    #    - For channel number and learning rate tuning, use (1,9). 
    #    - For epoch number tuning, use (10,13)
    #
    # Otherwise: "train" re-trains the new model. 
    
    RUN_MODE = "compare_models"  # Options: "train", "test_model", "compare_models"

    # IF "train" or "test_model" is selected, change MODEL_NUM to the model you want to test since these models are already pre-trained. Right now, 1-13
    MODEL_NUM = 1

    #### MAIN HYPERPARAMETERS FOR TUNING ####
    #
    # Each model has their own combination of hyperparameters.
    # This specific model is model [X].
    HyperParam_Channels = [32, 64, 128, 256]
    HyperParam_LR = 1e-3
    HyperParam_Epochs = 50

    # Settings for data loading and preprocessing.
    BATCH_SIZE = 128
    DOWNLOAD = True
    SIZE = 28

    # Training settings
    CLIP = 1
    CHECKPOINT_PATH, OUTPUT_FOLDER = GetModelPaths(MODEL_NUM)

    if RUN_MODE == "compare_models":
        # Read saved checkpoints and create a comparison table without retraining or loading data.
        CompareSavedModels(start_model=1, end_model=13)

    else:
        # Load the BloodMNIST dataset for training, validation and testing.
        # Also loads the number of classes and metadata information
        DataTRAIN_Set, DataVAL_Set, DataTEST_Set, class_num, input_channel, info = LoadDataBloodMNIST(batch_size=BATCH_SIZE, 
                                                                                                             download=DOWNLOAD, size=SIZE)

        if RUN_MODE == "train":
            # Run the model training segmentation
            RunTraining(DataTRAIN_Set=DataTRAIN_Set, DataVAL_Set=DataVAL_Set, DataTEST_Set=DataTEST_Set,
                                    class_num=class_num, channel_in=input_channel, info=info, num_channels=HyperParam_Channels, 
                                    learn_rate=HyperParam_LR, epoch_num=HyperParam_Epochs, clip=CLIP, checkpoint_path=CHECKPOINT_PATH, output_folder=OUTPUT_FOLDER)

        elif RUN_MODE == "test_model":
            # Just run the model testing for a fully trained model with hyperparameters defined in saved model entity
            RunTesting(checkpoint_path=CHECKPOINT_PATH, DataTEST_Set=DataTEST_Set, class_num=class_num, info=info, output_folder=OUTPUT_FOLDER)

        else:
            # Syntaxing error handling
            raise ValueError('RUN_MODE must be either "train", "test_model", or "compare_models"')
