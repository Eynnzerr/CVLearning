### Features
1. Abandon LRN (Used in AlexNet)
    > First, we note that using local response normalisation (A-LRN network) does not improve on the model A without any normalisation layers. We thus do not employ normalisation in the deeper architectures (B–E).
2. Use very small conv. kernel size instead of large ones in the first conv. layer
    > It is easy to see that a stack of two 3 × 3 conv. layers (without spatial pooling in between) has an effective receptive field of 5 × 5; three such layers have a 7 × 7 effective receptive field.

    benefits:
    - More non-linear rectification layers
    - Less parameters
![parameters](https://img-blog.csdnimg.cn/img_convert/8e6fb5f02c7d31a3c2740418a194e6b2.png)
3. **1 × 1** kernel size in configuration C
    > In one of the configurations we also utilise 1 × 1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity).

### Architecture
The authors prepared different set of configurations for the network as below:
![configs](https://img-blog.csdnimg.cn/img_convert/2c39549160dc0b18f4c55f047b0b374f.png)

### Implementation in PyTorch
To save work, I just realise A, B, D and E configurations, for the paper already claims that LRN is not good, and 1 * 1 kernel size in C is different from others', which adds complexity to codes.

```python
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
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

### Evaluation
Because of the GPU limitation of my PC, I can only train a DNN up to VGG11(otherwise my gpu memory will overflow), or VGG13 with tiny batch size(which is worse than training VGG11). So here I just trained a VGG11 on the same dataset which I trained AlexNet last time, with batch size set to be 64.
- Loss during training(20 epoch):
![result_loss.png](https://s2.loli.net/2022/06/19/ug7CWs9Bbk51FNI.png)
- Accuracy during training(20 epoch):
![result_accuracy.png](https://s2.loli.net/2022/06/19/pA7XT6LIneorthS.png)
Both with 100 samples.
And in validation set, it achieves 90% above accuracy.

### Conclusion
1. VGG proves that adding depth and more conv. layers to CNN, we may achieve better performance, the deeper, the better.(as long as overfit is not happening)
2. The computation resource of my laptop is so poor.