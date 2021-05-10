from torch.utils.data import Dataset
from xml.dom.minidom import parse
import numpy as np
from glob import glob
import os
import cv2
import torch
from utils.mytransform import *


# def xml2label1(xml_path):
#     dom_tree = parse(xml_path)
#     root_node = dom_tree.documentElement
#     indexs = ["xmin", "ymin", "xmax", "ymax"]
#     index_nodes = [root_node.getElementsByTagName(index) for index in indexs]
#     xmin = []
#     ymin = []
#     xmax = []
#     ymax = []
#     params = [xmin, ymin, xmax, ymax]
#     for cnt, index_node in enumerate(index_nodes):
#         for param in index_node:
#             params[cnt].append(param.childNodes[0].data)
#
#     h_node = root_node.getElementsByTagName("height")
#     height = next(iter(h_node)).childNodes[0].data
#     w_node = root_node.getElementsByTagName("width")
#     width = next(iter(w_node)).childNodes[0].data
#     loader = np.zeros(shape=[int(height), int(width)])
#     for x1, y1, x2, y2 in zip(xmin, ymin, xmax, ymax):
#         loader[int(y1):int(y2), int(x1):int(x2)] = 1
#     label_path = xml_path.replace(".xml", "_label.bmp").replace("\\", "/")
#
#     cv2.imwrite(label_path, loader * 255)
#     # cv2.imshow("a", loader)
#     # cv2.imwrite("1.bmp", loader*255)
#     # cv2.waitKey()
#     # exit()
#     return label_path

# 不同类别的rec可设置为不同灰度值输出
def xml2label(xml_path):
    dom_tree = parse(xml_path)
    root_node = dom_tree.documentElement
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    label_names = []
    params = [xmin, ymin, xmax, ymax, label_names]
    indexs = ["xmin", "ymin", "xmax", "ymax"]
    # 根据类别填充的灰度值
    fill_color = {"fire": 128, "smoke": 255}
    object_nodes = root_node.getElementsByTagName("object")
    for object_node in object_nodes:
        name_nodes = object_node.getElementsByTagName("name")
        label_name = next(iter(name_nodes)).childNodes[0].data
        params[4].append(label_name)
        for cnt, index in enumerate(indexs):
            rec_nodes = object_node.getElementsByTagName(index)
            params[cnt].append(next(iter(rec_nodes)).childNodes[0].data)
    h_node = root_node.getElementsByTagName("height")
    height = next(iter(h_node)).childNodes[0].data
    w_node = root_node.getElementsByTagName("width")
    width = next(iter(w_node)).childNodes[0].data
    # 生成原图大小的载体图
    loader = np.zeros(shape=[int(height), int(width)])
    for x1, y1, x2, y2, label_name in zip(xmin, ymin, xmax, ymax, label_names):
        loader[int(y1):int(y2), int(x1):int(x2)] = fill_color[label_name]
    label_path = xml_path.replace(".xml", "_label.bmp").replace("\\", "/")

    cv2.imwrite(label_path, loader)
    return label_path


# 加载数据集文件夹格式
#   data（1级目录）
# - image（2级目录）：存放图像数据
# - label（2级目录）：存放缺陷图像的标签图或者xml文件，若label中没有对应image的xml文件或者label图则会默认为良品
# 支持bmp、jpg格式图片
class Data_api_rec(Dataset):
    def __init__(self, root, gen_label=True, updata_txt=True, train_test_ratio=0.8, phase="test",
                 transform=None, Norm=None, use_relative_path=False):
        """

        :param root: 加载数据集的路径（数据集的格式为 image、label，其中label是存放缺陷图片的xml标注）
        :param gen_label: 是否根据xml生成对应的label图
        :param updata_txt: 是否更新txt文件
        :param train_test_ratio: 默认值为0.8,其中80%作为训练集和验证集，20%作为测试集,train_test_ratio=-1则获取image中的所有数据
        :param phase: 指定训练集的模式，in ["train", "test", "val"]
        :param transform: 数据增强部分，可以指定
        :param Norm: 数据归一化，先根据数据分布计算mean、std，外部指定
        :param use_relative_path: 在txt中保存的路径是否使用相对路径，要配合updata_txt使用
        """
        super(Data_api_rec, self).__init__()
        assert phase in ["train", "test", "val"]
        self.root = root
        self.phase = phase
        self.use_relative_path = use_relative_path
        self.classes_for_all_img = []
        self.get_phases_data = [self.phase]
        if train_test_ratio not in (0.0, 1.0):
            train_test_ratio = 1.0
            self.get_phases_data = ["train", "val", "test"]

        if gen_label:
            self.images_path = glob(os.path.join(self.root, "image", '*.*'))

            # image会被替换为label，所以其他文件夹名字不要出现image，否则会出现错误路径
            self.images_xml = [image_path.replace(".bmp", ".xml").replace(".jpg", ".xml").replace("image", "label")
                               for image_path in self.images_path]

            self.images_xml = [image_xml for image_xml in self.images_xml if os.path.exists(image_xml)]
            self.labels_path = [xml2label(image_xml) for image_xml in self.images_xml]

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
        images_path = glob(os.path.join(self.root, "image", '*.bmp'))
        images_path += glob(os.path.join(self.root, "image", '*.jpg'))
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
                f.write(val_list[i] + "\n")
        return images_path

    def get_image_list(self):
        all_phase_images = []
        for phase in self.get_phases_data:
            with open(os.path.join(self.root, phase + ".txt"), mode='r') as f:
                images = f.readlines()
            all_phase_images += [img.strip() for img in images]

        return all_phase_images

    # 平衡采样，给需要重复采样的数据进行人工打标，根据低对比度的路径名称分类打标签，
    # [0 1] 0 表示低对比度 1 表示正常
    def classes_for_all_imgs(self):
        for img_path in self.images_path:
            if img_path.split("_")[-1] == "low.bmp":
                class_id = 0
            else:
                class_id = 1
            # 对所有样本标签进行罗列
            self.classes_for_all_img.append(class_id)
        return self.classes_for_all_img

    def __getitem__(self, item):
        image_path = self.images_path[item % len(self.images_path)]
        image = cv2.imread(image_path)

        label_path = (image_path.split(".")[0] + "_label.bmp").replace("image", "label")

        image_array = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not os.path.exists(label_path):
            label_array = np.zeros([image_array.shape[0], image_array.shape[1]])
            image_label = torch.tensor([0])
        else:
            label_array = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            image_label = torch.tensor([1])
        image_tensor, label_tensor = self.trans([Image.fromarray(image_array), Image.fromarray(label_array)])

        if self.Norm:
            image_tensor = self.Norm(image_tensor)
        return image_tensor, label_tensor, image_path, image_label

    def __len__(self):
        return len(self.images_path)


def sampler_c(dataset):
    from torch.utils.data import sampler
    train_targets = dataset.classes_for_all_imgs()
    # 配置类别采样比例 4:1
    class_sample_couts = [4, 1]
    weights = torch.tensor(class_sample_couts, dtype=torch.float)
    # 为每一个样本采得 采样权重
    samples_weights = weights[train_targets]
    # 采样器 replacement=True-放回采样
    sampler = sampler.WeightedRandomSampler(weights=samples_weights, num_samples=len(dataset), replacement=True)
    return sampler


def main():
    root = "/media/root/软件/wqr/data/fire"
    dataset = Data_api_rec(root=root, gen_label=True, updata_txt=True,
                           train_test_ratio=-1, phase="train", use_relative_path=False)
    a = dataset[0]
    # sampler = sampler_c(dataset)
    # train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=False, sampler=sampler)

    # weights = [2 if label == 1 else 1 for _, _, _, label in dataset]
    # print(weights)
    # from data_api.get_dataset_mean_std import get_mean_std
    # a, b, _ = dataset[0]
    # mean, std = get_mean_std(dataset, ratio=0.1)
    # # print(mean)
    # # print(std)
    # # exit()
    # # print(a.shape)
    # # print(b.shape)
    # from torchvision import transforms
    # tran = transforms.ToPILImage()
    # imga = tran(a)
    # imgb = tran(b)
    # imgaa = cv2.cvtColor(np.array(imga), cv2.COLOR_RGB2BGR)
    #
    # cv2.imshow("a", np.uint8(imgaa))
    # cv2.imshow("b", np.uint8(imgb))
    # cv2.waitKey()


if __name__ == '__main__':
    main()

