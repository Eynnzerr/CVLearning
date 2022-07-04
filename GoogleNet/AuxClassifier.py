import torch
import torch.nn as nn
import torch.nn.functional as F
from ReluConv import ReluConv


class AuxClassifier(nn.Module):

    def __init__(self, ch_in, num_classes=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ReluConv(ch_in=ch_in, ch_out=128, kernel_size=1, stride=1)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)  # input is N x 128 x 4 x 4
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)  # as input of fc layer
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x
