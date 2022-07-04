from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):

    # 从脚本生成的txt文件读取图片数据和标签，以元组形式存入imgs列表中
    def __init__(self, txt_path, transform=None):
        imgs = []
        with open(txt_path, 'r') as data:
            for line in data:
                line = line.rstrip()
                info = line.split()
                imgs.append((info[0], int(info[1])))
                self.imgs = imgs
                self.transform = transform

    # 每次迭代返回下一张图片数据及其标签
    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)