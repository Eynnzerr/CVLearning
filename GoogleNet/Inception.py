import torch
import torch.nn as nn
from ReluConv import ReluConv


class Inception(nn.Module):

    def __init__(self, ch_in, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        self.parallel1 = ReluConv(ch_in=ch_in, ch_out=ch1x1, kernel_size=1, stride=1)
        self.parallel2 = nn.Sequential(
            ReluConv(
                ch_in=ch_in,
                ch_out=ch3x3_reduce,
                kernel_size=1,
                stride=1
            ),
            ReluConv(
                ch_in=ch3x3_reduce,
                ch_out=ch3x3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.parallel3 = nn.Sequential(
            ReluConv(
                ch_in=ch_in,
                ch_out=ch5x5_reduce,
                kernel_size=1,
                stride=1
            ),
            ReluConv(
                ch_in=ch5x5_reduce,
                ch_out=ch5x5,
                kernel_size=5,
                stride=1,
                padding=2
            )
        )
        self.parallel4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ReluConv(
                ch_in=ch_in,
                ch_out=pool_proj,
                kernel_size=1,
                stride=1
            )
        )

    def forward(self, x):
        # Do DepthConcat
        parallel1 = self.parallel1(x)
        parallel2 = self.parallel2(x)
        parallel3 = self.parallel3(x)
        parallel4 = self.parallel4(x)
        return torch.cat((parallel1, parallel2, parallel3, parallel4), 1)
