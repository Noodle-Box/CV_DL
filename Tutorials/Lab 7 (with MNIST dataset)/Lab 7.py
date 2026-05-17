## Tutorial 7: Generative Model with MNIST dataset

# Importing Libraries
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


####### CUDA / GPU SETUP ########

# Use NVIDIA CUDA GPU if PyTorch can access it.
# Otherwise, fall back to CPU so the script still runs.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
else:
    print("CUDA is not available. PyTorch is currently using CPU.")

print("Loading local MNIST dataset...")
####### Task 1: Load MNIST dataset from torch library ########

# The MNIST has 70,000 images.
# 60,000 training images and 10,000 testing images
# Each image is 28x28 pixels black and white/grayscale
# 10 possible classes: digits 0-9
# 
# Each sample in the dataset is a tuple (image, label) where:
# - image: a 28x28 pixel grayscale image represented as a PyTorch tensor
# - label: an integer representing the digit (0-9) corresponding to the image

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")

# Convert images to PyTorch tensors and normalize them
transform = transforms.ToTensor()

# Download the training dataset. From https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
train_dataset = datasets.MNIST(
    root=data_path, 
    train=True,      # Set to True for training dataset
    transform=transform, 
    download=False)  # Set to False if already downloaded, True to download again

# Download the testing dataset
test_dataset = datasets.MNIST(
    root=data_path, 
    train=False,     # Set to False for test dataset
    transform=transform, 
    download=False)  # Set to False if already downloaded, True to download again

print("MNIST downloaded successfully.")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Optional: create DataLoaders for training later
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=(device.type == "cuda")  # Speeds up CPU-to-GPU transfer when using CUDA
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)

# Check one batch
images, labels = next(iter(train_loader))

# Move batch to the selected device.
# If CUDA is available, these tensors will be moved to your NVIDIA GPU.
images = images.to(device)
labels = labels.to(device)

print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")
print(f"Images are stored on: {images.device}")
print(f"Labels are stored on: {labels.device}")


############ Task 2: Defining the Neural Network ##########

# Source: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# Initialize NN class and define the layers in the __init__ method. This NN has 3 fully connected layers with ReLU activation functions in between. 
# The input layer takes the 28x28 pixel images (flattened to 784)
# The output layer has 10 neurons corresponding to the 10 digit classes (0-9).
#
# Right now the model is untrained at this phase so weights are randomized.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Converts batch of 28x28 images into a batch of 784-dimensional vectors (1*28*28 = 784)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 784 inputs -> 512 outputs
            nn.ReLU(),             # Activation function to introduce non-linearity
            nn.Linear(512, 512),   # 512 hidden neurons -> 512 hidden neurons
            nn.ReLU(),             # Activation function to introduce non-linearity
            nn.Linear(512, 10),    # 512 hidden neurons -> 10 digit classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Create an instance of the neural network and move it to the selected device (GPU or CPU).
model = NeuralNetwork().to(device)
print(model)
print(f"Model is stored on: {next(model.parameters()).device}")


####### Test one forward pass ########

# Get one batch from the training loader
images, labels = next(iter(train_loader))

# Move images and labels to GPU/CPU
images = images.to(device)
labels = labels.to(device)

# Pass images through the model
logits = model(images)

print(f"\nInput image batch shape: {images.shape}")
print(f"Output logits shape: {logits.shape}")

# Convert logits to probabilities
pred_probab = nn.Softmax(dim=1)(logits)

# Get predicted digit class
y_pred = pred_probab.argmax(1)

print(f"Predicted labels: {y_pred[:10]}")
print(f"Actual labels:    {labels[:10]}")




########## Task 3: Training the NN with 10-15 epochs ###########

