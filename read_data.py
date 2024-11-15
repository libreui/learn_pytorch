from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        """获取数据集中每个数据"""
        self.img_name = self.img_path[idx]
        self.img_item_path = os.path.join(self.root_dir, self.label_dir, self.img_name)
        img = Image.open(self.img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        """获取数据集中数据的个数"""
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = "dataset/hymenoptera_data/train"
    ants_label_dir = "ants"
    bees_label_dir = "bees"
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)

    # 训练数据集
    train_dataset = ants_dataset + bees_dataset
    img, lable = train_dataset[124]
    img.show()
