import torch.nn as nn

configs = {
    'A': [64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
    'B': [64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1],
    'D': [64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1],
    'E': [64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1]
}


class VGG(nn.Module):

    def __init__(self, net_type, num_classes=2, init_weights=False):
        super().__init__()
        # Construct conv. layers using given configuration
        assert net_type in configs, 'Error: type {} is not supported currently.'.format(net_type)
        config = configs[net_type]
        layers = []
        in_c = 3
        for c in config:
            if c == -1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_channels=in_c, out_channels=c, kernel_size=3, padding=1), nn.ReLU()]
                in_c = c
        self.conv = nn.Sequential(*layers)

        # Initialize weights
        if init_weights:
            self._init_weights()

        # FC layers are the same for all configurations
        self.fc = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=500, out_features=20),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=20, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512*7*7)
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
