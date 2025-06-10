import torch.nn as nn
import torch.nn.functional as F

class RectangleDetector(nn.Module):
    def __init__(self):
        super(RectangleDetector, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.pool is reused
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.pool is reused

        # After 3 pooling layers of stride 2, 64x64 becomes 8x8
        # Flattened size: 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 4) # Output layer for 4 coordinates

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps
        x = x.view(-1, 64 * 8 * 8) 
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No activation, as this is a regression output
        return x

model = RectangleDetector()
print(model)