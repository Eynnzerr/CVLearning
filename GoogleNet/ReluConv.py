import torch.nn as nn


# Basic conv layer with ReLU activation.
class ReluConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
