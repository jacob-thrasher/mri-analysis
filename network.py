import torch
from torch import nn

class MRICNN(nn.Module):
    def __init__(self, in_ch=3, min_features=32, depth=4, kernel_size=3, n_class=3):
        super().__init__()

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_ch, min_features, kernel_size=kernel_size))
        for d in range(depth-1):
            conv_layers.append(nn.Conv2d(min_features*(2**d), 
                                         min_features*(2**(d+1)),
                                         kernel_size=kernel_size))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, n_class), # TODO: Make input variable based on params
        )

        
    def forward(self, x):
        for layer in self.conv_layers:
            x = self.pool(self.relu(layer(x)))

        return nn.functional.softmax(self.head(x), dim=1)
        