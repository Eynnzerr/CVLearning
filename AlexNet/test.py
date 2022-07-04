from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('alexnet-catvsdog.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式

    trans = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    classes = ('cat', 'dog')

    iter_time = 100
    for i in range(1, iter_time + 1):
        # 读取要预测的图片
        # 读取要预测的图片
        img = Image.open(
            f"/home/eynnzerr/open/pyTorch-datasets/dogs-vs-cats-redux-kernels-edition/test/%d.jpg" % i)  # 读取图像
        # img.show()
        # plt.imshow(img)  # 显示图片
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()

        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]

        # 预测
        # 预测
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率
        print(f'第%d张图片:' % i)
        print("概率：", prob)
        value, predicted = torch.max(output.data, 1)
        predict = output.argmax(dim=1)
        pred_class = classes[predicted.item()]
        print("预测类别：", pred_class)
