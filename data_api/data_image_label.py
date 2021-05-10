from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os
import cv2
import torch
from utils.mytransform import *


class Data_image_label(Dataset):
    def __init__(self, root, updata_txt=True, train_test_ratio=0.8, phase="test",
                 transform=None, Norm=None, use_relative_path=False):
        """
        :param root: 加载数据集的路径（数据集的格式为 image、label，其中label是存放缺陷图片的xml标注）
        :param gen_label: 是否根据xml生成对应的label图
        :param updata_txt: 是否更新txt文件
        :param train_test_ratio: 默认值为0.8,其中80%作为训练集和验证集，20%作为测试集
        :param phase: 指定训练集的模式，in ["train", "test", "val"]
        :param transform: 数据增强部分，可以指定
        :param Norm: 数据归一化，先根据数据分布计算mean、std，外部指定
        :param use_relative_path: 在txt中保存的路径是否使用相对路径，要配合updata_txt使用
        """
        super(Data_image_label, self).__init__()
        assert phase in ["train", "test", "val"]
        self.root = root
        self.phase = phase
        self.use_relative_path = use_relative_path
        self.P_images_path = []
        self.N_images_path = []
        if not os.path.exists(os.path.join(self.root, "train.txt")) or updata_txt:
            self.images_path = self.make_txt(train_test_ratio=train_test_ratio)
        self.images_path = self.get_image_list()

        self.Norm = Norm
        if not transform:
            if self.phase == "train":
                self.trans = GroupCompose([GroupResize([128, 768]),
                                           GroupRandomHorizontalFlip(p=0.5),
                                           GroupRandomVerticalFlip(p=0.5),
                                           GroupRandomHorizontalMove(),
                                           GroupToTensor()])
            else:
                self.trans = GroupCompose([GroupResize([128, 768]),
                                           GroupToTensor()])
        else:
            self.trans = transform

    def make_txt(self, train_test_ratio=0.5):
        self.P_images_path = glob(os.path.join(self.root, '良品/*.bmp'))
        self.N_images_path = glob(os.path.join(self.root, '缺陷/*.bmp'))
        images_path = self.P_images_path + self.N_images_path
        if self.use_relative_path:
            dataset_name = self.root.split("/")[-1]
            images_path = [dataset_name + image_path.split(dataset_name)[1] for image_path in images_path]
        random.shuffle(images_path)
        seg_point = int(len(images_path) * train_test_ratio)
        train_list = images_path[0:int(0.8*seg_point)]
        test_list = images_path[seg_point:len(images_path)]
        val_list = images_path[int(0.8*seg_point):int(seg_point)]
        with open(os.path.join(self.root, "train.txt"), mode='w') as f:
            for i in range(len(train_list)):
                f.write(train_list[i] + "\n")
        with open(os.path.join(self.root, "test.txt"), mode='w') as f:
            for i in range(len(test_list)):
                f.write(test_list[i] + "\n")
        with open(os.path.join(self.root, "val.txt"), mode='w') as f:
            for i in range(len(val_list)):
                f.write(test_list[i] + "\n")
        return images_path

    def get_image_list(self):
        with open(os.path.join(self.root, self.phase + ".txt"), mode='r') as f:
            images = f.readlines()
        images = [img.strip() for img in images]
        return images

    def __getitem__(self, item):
        image_path = self.images_path[item % len(self.images_path)]
        image = cv2.imread(image_path)
        image_array = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = self.trans(Image.fromarray(image_array))
        # 根据路径名称进行类别赋值
        if image_path.split("/")[-2] == "缺陷":
            image_label = torch.tensor(1)
        else:
            image_label = torch.tensor(0)
        if self.Norm:
            image_tensor = self.Norm(image_tensor)
        return image_tensor, image_tensor, image_path, image_label,

    def __len__(self):
        return len(self.images_path)


if __name__ == "__main__":
    dataset = Data_image_label(root="/media/root/软件/wqr/data/groove", updata_txt=True)
    a, b = dataset[1]
    print(a.shape)
    print(b)


