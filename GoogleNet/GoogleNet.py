import torch
import torch.nn as nn
from ReluConv import ReluConv
from Inception import Inception
from AuxClassifier import AuxClassifier


class GoogleNet(nn.Module):

    def __init__(self, num_classes=2, aux_enabled=True):
        super().__init__()
        '''
        input: 227 x 227 x 3
        output: 56 x 56 x 64
        '''
        self.conv1 = ReluConv(ch_in=3, ch_out=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # LRN?

        '''
        input: 56 x 56 x 64
        output: 28 x 28 x 192
        '''
        self.conv2 = ReluConv(ch_in=64, ch_out=64, kernel_size=1, stride=1)
        self.conv3 = ReluConv(ch_in=64, ch_out=192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # LRN?

        '''
        input: 28 x 28 x 192
        output: 28 x 28 x 256  
        '''
        self.inception3a = Inception(
            ch_in=192,
            ch1x1=64,
            ch3x3_reduce=96,
            ch3x3=128,
            ch5x5_reduce=16,
            ch5x5=32,
            pool_proj=32
        )

        '''
        input: 28 x 28 x 256
        output: 14 x 14 x 480  
        '''
        self.inception3b = Inception(
            ch_in=256,
            ch1x1=128,
            ch3x3_reduce=128,
            ch3x3=192,
            ch5x5_reduce=32,
            ch5x5=96,
            pool_proj=64
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        '''
        input: 14 x 14 x 480
        output: 14 x 14 x 512  
        '''
        self.inception4a = Inception(
            ch_in=480,
            ch1x1=192,
            ch3x3_reduce=96,
            ch3x3=208,
            ch5x5_reduce=16,
            ch5x5=48,
            pool_proj=64
        )

        '''
        input: 14 x 14 x 512
        output: 14 x 14 x 512  
        '''
        self.inception4b = Inception(
            ch_in=512,
            ch1x1=160,
            ch3x3_reduce=112,
            ch3x3=224,
            ch5x5_reduce=24,
            ch5x5=64,
            pool_proj=64
        )

        '''
        input: 14 x 14 x 512
        output: 14 x 14 x 512  
        '''
        self.inception4c = Inception(
            ch_in=512,
            ch1x1=128,
            ch3x3_reduce=128,
            ch3x3=256,
            ch5x5_reduce=24,
            ch5x5=64,
            pool_proj=64
        )

        '''
        input: 14 x 14 x 512
        output: 14 x 14 x 528  
        '''
        self.inception4d = Inception(
            ch_in=512,
            ch1x1=112,
            ch3x3_reduce=144,
            ch3x3=288,
            ch5x5_reduce=32,
            ch5x5=64,
            pool_proj=64
        )

        '''
        input: 14 x 14 x 528
        output: 14 x 14 x 832  
        '''
        self.inception4e = Inception(
            ch_in=528,
            ch1x1=256,
            ch3x3_reduce=160,
            ch3x3=320,
            ch5x5_reduce=32,
            ch5x5=128,
            pool_proj=128
        )

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        '''
        input: 7 x 7 x 832
        output: 7 x 7 x 832  
        '''
        self.inception5a = Inception(
            ch_in=832,
            ch1x1=256,
            ch3x3_reduce=160,
            ch3x3=320,
            ch5x5_reduce=32,
            ch5x5=128,
            pool_proj=128
        )

        '''
        input: 7 x 7 x 832
        output: 7 x 7 x 1024  
        '''
        self.inception5b = Inception(
            ch_in=832,
            ch1x1=384,
            ch3x3_reduce=192,
            ch3x3=384,
            ch5x5_reduce=48,
            ch5x5=128,
            pool_proj=128
        )

        '''
        input: 7 x 7 x 1024
        output: 1 x 1 x 1024  
        '''
        self.pool5 = nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=True)
        self.dropout = nn.Dropout(0.4)

        '''
        input: 1024
        output: 2  
        '''
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

        if aux_enabled:
            self.aux_enabled = True
            self.aux0 = AuxClassifier(ch_in=512, num_classes=2)
            self.aux1 = AuxClassifier(ch_in=528, num_classes=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        if self.training and self.aux_enabled:
            aux0 = self.aux0(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_enabled:
            aux1 = self.aux1(x)

        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_enabled:
            return x, aux0, aux1
        return x
