import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module): # line 1
    def __init__(self):
        super(Network, self).__init__() # line 3
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.Dropout2d(0,3),
            nn.MaxPool2d(2)
        )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(),
            nn.Dropout2d(0,3),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, input):
        input = self.conv_layer1(input)
        input = self.conv_layer2(input)
        input = input.reshape([-1, 12 * 4 * 4])
        input = self.fc1(input)
        input = self.fc2(input)
        input = F.softmax(self.out(input))
        return input

