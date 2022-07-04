import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyDataset import MyDataset

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

txt_path = '/home/eynnzerr/open/pyTorch-datasets/dogs-vs-cats-redux-kernels-edition/train.txt'

train_data = MyDataset(txt_path=txt_path, transform=transform_train)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

classes = ('cat', 'dog')

if __name__ == '__main__':
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_label) = next(examples)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        img = example_data[i]
        img = img.numpy()  # FloatTensor转为ndarray
        img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        plt.imshow(img)
        plt.title("label:{}".format(example_label[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
