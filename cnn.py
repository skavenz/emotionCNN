import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnnLayer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnnLayer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnnLayer3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)  # [(W−K+2P)/S]+1
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 8)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()

    def forward(self, out):
        out = torch.relu(self.cnnLayer1(out))
        out = self.pool(out)

        out = torch.relu(self.cnnLayer2(out))
        out = self.pool(out)

        out = torch.relu(self.cnnLayer3(out))
        out = self.pool(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = torch.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out
