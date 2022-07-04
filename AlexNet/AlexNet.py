import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # The first convolutional layer
            # input: 227 × 227 × 3 (original image)
            # output: 27 × 27 × 96
            # 96 kernels of size 11 × 11 × 3  stride: 4
            # with LRN and overlapping pooling
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # The second convolutional layer
            # input: 27 × 27 × 96
            # output: 13 × 13 × 256
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # The third convolutional layer
            # input: 13 × 13 × 256
            # output: 13 × 13 × 384
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            # The fourth convolutional layer
            # input: 13 × 13 × 384
            # output: 13 × 13 × 384
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),

            # The fifth convolutional layer
            # input: 13 × 13 × 384
            # output: 6 × 6 × 256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Parameters changed slightly since we use different dataset
        self.fc = nn.Sequential(
            # The first fully connected layer
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(6 * 6 * 256), out_features=500),
            nn.ReLU(),

            # The second fully connected layer
            nn.Dropout(p=0.5),
            nn.Linear(in_features=500, out_features=20),
            nn.ReLU(),

            # The third fully connected layer i.e. softmax classifier
            nn.Linear(in_features=20, out_features=2)
        )

    def forward(self, x):
        conv_result = self.conv(x)
        result = conv_result.view(-1, 6 * 6 * 256)  # flattened to vector
        return self.fc(result)
