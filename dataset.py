from torchvision import transforms
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torch.utils.data import DataLoader
from utils.mytrans import GroupRandomVerticalFlip, GroupRandomHorizontalFlip, GroupCompose


class Image_data(Dataset):
    def __init__(self, root, mode="train"):
        super(Image_data, self).__init__()
        self.transform = transforms.Compose([transforms.Resize([128, 384]), transforms.ToTensor()])
        self.img = []
        self.mode = mode
        self.root = root
        self.Group_trans = GroupCompose([GroupRandomVerticalFlip(p=0.5), GroupRandomHorizontalFlip(p=0.5)])
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.img += glob.glob(os.path.join(self.root, name, '*.bmp'))

        self.img_label_path = [path for path in self.img if '_label' in path]

        if mode == 'train':
            # self.img_label_path = self.img_label_path[0:len(self.img_label_path)-30]
            self.img_label_path = self.img_label_path[0:len(self.img_label_path)]
        if mode == 'test':
            # self.img_label_path = self.img_label_path[len(self.img_label_path)-30:len(self.img_label_path)]
            self.img_label_path = self.img_label_path[0 :len(self.img_label_path)]

    def __getitem__(self, index):
        img_label_path = self.img_label_path[index % len(self.img_label_path)]
        img_path = img_label_path.replace('_label', '')
        img_gray = Image.open(img_path)
        img_label = Image.open(img_label_path)
        if self.mode == "train":
            [img_gray, img_label] = self.Group_trans([img_gray, img_label])
        img_gray_tensor = self.transform(img_gray)
        img_label_tensor = self.transform(img_label)
        return img_gray_tensor, img_label_tensor, img_label_path

    def __len__(self):
        return len(self.img_label_path)


def main():
    db = Image_data('/media/root/文档/LZY/第四步_像素级标注/划痕/划痕9.02+8.28train')
    a, b = db[0]
    # loader = DataLoader(db, batch_size=1, shuffle=True)
    # a, b = next(iter(loader))


if __name__ == '__main__':
    main()

