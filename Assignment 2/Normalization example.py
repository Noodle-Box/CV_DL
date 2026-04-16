import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # 1. Batch Normalization Layer
        # It needs the number of channels (16) as an input
        self.bn1 = nn.BatchNorm2d(16)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # The standard sequence: Conv -> BatchNorm -> Activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# Example usage:
model = SimpleCNN()
dummy_input = torch.randn(8, 3, 32, 32) # Batch size of 8
output = model(dummy_input)

print(f"Output shape: {output.shape}")