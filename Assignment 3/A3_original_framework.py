import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import medmnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from medmnist import INFO


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ------------------ From here, we build our neural network model --------------------
# First, we build the residual blocks. The following codes build Residual Block-I and Residual Block-II (feel free to replace this part with your own codes).
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


# ---------------- Then we build the ResNet structure (YOU NEED TO COMPLETE THIS PART) ----------------
class ResNet18(nn.Module):

    ########### BUILD YOUR CODE HERE ############


# ---------------- From here, we define a function to train the neural network (YOU NEED TO COMPLETE THIS PART) ----------------
def train(model, train_dataset, optimizer, criterion, clip):
    model.train()

    optimizer.zero_grad()       # Initialize the neural network gradient

    ########### BUILD YOUR CODE HERE ############


# ---------------- From here, we define a function to evaluate the trained neural network (YOU NEED TO COMPLETE THIS PART) ----------------
def evaluate(model, test_dataset, criterion):
    model.eval()

    ########### BUILD YOUR CODE HERE ############


# This is the main function
if __name__ == '__main__':
    data_type = 'bloodmnist'  # We use the 2D BloodMNIST dataset in our task (images obtained from the blood cell microscope)

    info = INFO[data_type]  # Extract the information of this dataset

    n_classes = len(info['label'])  # Extract the number of classes in this dataset
    input_channel = info['n_channels']     # Extract the number of channels in each image sample (BloodMNIST has images with 3 color channels)

    DataClass = getattr(medmnist, info['python_class'])

    # Define a data preprocessing pipline, to make all the images with a mean value of 0.5 and a deviation of 0.5
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    # Load the data (Note that we have  not loaded the validation data in current code framework. You will need to use the validation data to select the optimum hyperparameters)
    train_data = DataClass(split='train', transform=data_transform, download=False)  # Set download=True if you cannot load the data successfully
    test_data = DataClass(split='test', transform=data_transform, download=False)  # Set download=True if you cannot load the data successfully

    # We plot the information of the training and testing data (Read them in detail if you want to gain more biological information of the datasetS)
    print(train_data)
    print('------------------------------------------------------------------------------------------------------------------------------------\n')
    print(test_data)
    print('------------------------------------------------------------------------------------------------------------------------------------\n')

    # Visualize the sample image
    sample_idx = 534                    # Visualize the 534th sample image in the training dataset
    sample = train_data[sample_idx]     # Get the image tensor
    sample_image = sample[0]            # Get the image
    sample_label = sample[1]            # Get the label of this image
    plt.imshow(np.abs(sample_image.permute(1, 2, 0)))       # Plot the image

    # Put the training and testing datasets in dataloader structures. You will use the dataloaders in your training and testing functions to visit each sample of the batch.
    BATCH_SIZE = 128  # Set the batch size to 128 (i.e., 128 samples in a batch)
    train_dataset = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # shuffle=True means we shuffle all the 128 samples in each training epoch
    test_dataset = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)  # We perform the testing process one sample by one sample, so batch_size=1. We do not need to shuffle the testing cases, so shuffle=False

    # Number of channels (i.e., C1, C2, C3, C4 in Figure 1 of the assignment task sheet). This is the hyperparameter that you need to determine
    # YOU NEED TO FIND THE OPTIMUM NUMBER OF CHANNELS IN YOUR ASSIGNMENT
    # Hint: Consider the three possible sets [8, 16, 32, 64], [32, 64, 128, 256], and [64, 128, 256, 512]
    channel_nums = [16, 32, 64, 128]

    # Create an instance of the neural network, and then put it in the CPU or GPU. Note that you may need to pass some parameters to ResNet18() if your construction function of ResNet18() need to use them
    model = ResNet18().to(device)

    # We use CrossEntropyLoss function in our multi-class task
    criterion = nn.CrossEntropyLoss()

    # We clip the gradient to avoid the gradient explosion
    # Use torch.nn.utils.clip_grad_norm_(model.parameters(), clip) in your training function to implement this gradient clip (by setting clip=1, we clip the gradient during the training)
    CLIP = 1

    # Learning rate is initially set as 1e-4. This is another hyperparameter that you need to determine
    # YOU NEED TO FIND THE OPTIMUM LEARNING RATE IN YOUR ASSIGNMENT
    # Hint: Consider 1e-3 and 1e-5 in addition to the initial one
    learning_rate = 1e-4

    # We use ADAM as our optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # The maximum number of epochs to train the neural network. The initial maximum number of epochs is set as 100 (you may increase it if you feel 100 is not a sufficient number for achieving the optimum performance).
    # YOU NEED TO FIND THE OPTIMUM NUMBER OF EPOCHS THAT GIVES THE BEST CLASSIFICATION PERFORMANCE.
    epoch_num = 100

    # ---------------- From here, we enter into the training epoch (YOU NEED TO COMPLETE THIS PART) ----------------
    for epoch in range(0, epoch_num):

    ########### BUILD YOUR CODE HERE ############


    # ---------------- From here, we evaluate the trained neural network (YOU NEED TO COMPLETE THIS PART) ----------------
    # The assessors should be able to load and evaluate the trained models here

    ########### BUILD YOUR CODE HERE ############
