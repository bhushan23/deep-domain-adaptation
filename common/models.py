import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSimpleFeatureNet(nn.Module):
    def __init__(self):
        super(BaseSimpleFeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 4)
        self.maxpool1 = nn.MaxPool2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.maxpool2 = nn.MaxPool2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.maxpool3 = nn.MaxPool2d(64)
        self.fc1 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x = self.maxpool2(x)
        x = F.relu(x)
        #x = self.conv3(x)
        #x = self.maxpool3(x)
        #x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 13)

    def forward(self, x):
        #print x.shape
        x = self.fc1(x)
        x = F.relu(x)
        #print x.shape
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        return x
