################################################# Import Libraries ################################################
# Standard imports
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

############################################# Neural Network Modelling, Training, Testing ##############################################


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
        out = self.initial(x)
        out = self.block_cascade(out)
        out = self.global_avg_pool(out)
        out = self.flatten(out)
        out = self.fully_connected(out)
        return out


# ---------------- From here, we define a function to train the neural network (YOU NEED TO COMPLETE THIS PART) ----------------
def train(model, train_dataset, optimizer, criterion, clip):
    model.train()

    optimizer.zero_grad()       # Initialize the neural network gradient

    ########### BUILD YOUR CODE HERE ############


    ## Why do we use crossEntropyLoss?
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is used for multi-class classification tasks. 
    clip = clip
    training_data = train_dataset

# ---------------- From here, we define a function to evaluate the trained neural network (YOU NEED TO COMPLETE THIS PART) ----------------
def evaluate(model, test_dataset, criterion):
    model.eval()

    ## Why do we use crossEntropyLoss?
    ########### BUILD YOUR CODE HERE ############

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

    # Confirm datasets
    print(train_data)
    print('------------------------------------------------------------------------------------------------------------------------------------\n')
    print(validate_data)
    print('------------------------------------------------------------------------------------------------------------------------------------\n')
    print(test_data)
    print('------------------------------------------------------------------------------------------------------------------------------------\n')

    # Put the training and testing datasets in dataloader structures. 
    # You will use the dataloaders in your training and testing functions to visit each sample of the batch.
    pin_memory = device.type == "cuda"
    train_dataset = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    validate_dataset = data.DataLoader(dataset=validate_data, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_dataset = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

    return train_dataset, validate_dataset, test_dataset, n_classes, input_channel, info


########################################## Main Function for Data Loading, Training, and Testing ##########################################


# This is the main function
if __name__ == '__main__':

    BATCH_SIZE = 128  # Set the batch size to 128 (i.e., 128 samples in a batch)
    DOWNLOAD = True   # Set download to True to download the dataset if you don't have it in your local directory
    SIZE = 28         # Set the image size to 28 (i.e., 28x28 pixels)
    train_dataset, validate_dataset, test_dataset, n_classes, input_channel, info = load_bloodmnist_data(batch_size=BATCH_SIZE, download=DOWNLOAD, size=SIZE)

    ########################################################## ######################################################################

    # Number of channels (i.e., C1, C2, C3, C4 in Figure 1 of the assignment task sheet). This is the hyperparameter that you need to determine
    # YOU NEED TO FIND THE OPTIMUM NUMBER OF CHANNELS IN YOUR ASSIGNMENT
    # Hint: Consider the three possible sets [8, 16, 32, 64], [32, 64, 128, 256], and [64, 128, 256, 512]
    channel_nums = [16, 32, 64, 128]

    # Create an instance of the neural network, and put it on GPU.
    # Note for markers: I have an RTX5070 GPU so I'll be training it locally
    # channel_nums is passed into ResNet18 so different [C1, C2, C3, C4] settings can be tested as hyperparameter experiments.
    model = ResNet18(input_channels=input_channel, channel_nums=channel_nums, n_classes=n_classes).to(device)

    # We use CrossEntropyLoss function in our multi-class task
    criterion = nn.CrossEntropyLoss()

    # We clip the gradient to avoid the gradient explosion
    # Use torch.nn.utils.clip_grad_norm_(model.parameters(), clip) in your training function to implement this gradient clip (by setting clip=1, we clip the gradient during the training)
    CLIP = 1

    ########################################################## ######################################################################

    # Learning rate is initially set as 1e-4. This is another hyperparameter that you need to determine
    # YOU NEED TO FIND THE OPTIMUM LEARNING RATE IN YOUR ASSIGNMENT
    # Hint: Consider 1e-3 and 1e-5 in addition to the initial one
    learning_rate = 1e-4


    ########################################################## ######################################################################
    # We use ADAM as our optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    ########################################################## ######################################################################    # The maximum number of epochs to train the neural network. The initial maximum number of epochs is set as 100 (you may increase it if you feel 100 is not a sufficient number for achieving the optimum performance).
    # YOU NEED TO FIND THE OPTIMUM NUMBER OF EPOCHS THAT GIVES THE BEST CLASSIFICATION PERFORMANCE.
    epoch_num = 10

    # ---------------- From here, we enter into the training epoch (YOU NEED TO COMPLETE THIS PART) ----------------
    for epoch in range(0, epoch_num):

    ########### BUILD YOUR CODE HERE ############
        pass


########################################################## ######################################################################
    # ---------------- From here, we evaluate the trained neural network (YOU NEED TO COMPLETE THIS PART) ----------------
    # The assessors should be able to load and evaluate the trained models here

    ########### BUILD YOUR CODE HERE ############
    # evaluate()
