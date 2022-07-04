### Architecture
#### techniques
1. Non-linear activation function: ReLU
2. GPU: two GTX-580, each with 3 GB memory
3. Overlapping Pooling: stride(2) != kernel size(3): More diffcult to overfit
4. Local Response Normalization: Aids generalization
5. Data augmentation:
   - generating image translations and horizontal reflections
   - altering the intensities of the RGB channels in training images(using **PCA**)
6. Dropout:
Setting to zero the output of each hidden neuron with probability 0.5. The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in back-propagation. So every time an input is presented, the neural network samples a different architecture, but all these architectures share weights. This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.
#### overall architecture
5 convolutional layers and 3 fully connected layers(including a softmax classifier). 
Pytorch version is as follows.
```python
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
```
**Note**: To replicate the network easier, I use a much smaller dataset on Kaggle instead, and thus modify a few parameters of the fully connected layers.

### evaluation
After 20 epoches of training(without validating), plot the loss curve and accuracy curve as below.
![alexnet_res1.png](https://s2.loli.net/2022/06/16/yJqNAO7c1vXb3go.png)
![alexnet_res2.png](https://s2.loli.net/2022/06/16/Y8x6WbB7hAtdvky.png)
It's notable that the loss curve tends to shake heavily, which is unexpectable. I suppose this is because of the following reasons:
- The normalization parameters used in the paper is not suitable for the dataset I used.
- The dataset is still too small for the large deep CNN.
- The evaluation function I used (Cross-entropy) may not be propriate.

Using the model trained from above to detect the picture(not in training set) below:
![alexnet_res3.png](https://s2.loli.net/2022/06/16/kp56a3MjWYEcgHs.png)
Results:
probability： cat: 9.9946e-01, dog: 5.4484e-04
predict class： cat

### summary
AlexNet first approved that a large, deep convolutional neural network is capable of achieving record-breaking results on a highly challenging dataset using purely supervised learning. However, It is notable that the network’s performance degrades if a single convolutional layer is removed.
AlexNet proposed a new and powerful activation function, i.e. ReLU, which is better than sigmoid and tanh most of the time.
AlexNet brings dropout algorithm to good use.