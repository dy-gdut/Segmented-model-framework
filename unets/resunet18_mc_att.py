from unets.unet_blocks import *
from unets.resnet_blocks import _resnet, BasicBlock, Bottleneck
import torch
from torch import nn

"""
1. resnet_net 采用了5个不同尺度的特征图图  level：5
2. 用三个3*3卷积代替 7*7卷积，并且步长全部为1,得到与原始图片尺寸相同的特征
3. base_channels控制着网络的宽度
4.   stride：1   网络输出与输入尺寸相同
"""


class Res18_UNet(UNet):
    def __init__(self, n_classes, norm_layer=None, bilinear=True, pretrained=False, **kwargs):
        self.base_channels = kwargs.get("base_channels", 32)  # resnet18 和resnet34 这里为 32 , 64
        self.pretrained = pretrained
        level = kwargs.get("level", 5)
        self.b_RGB = kwargs.get("b_RGB", True)

        padding = 1
        super(Res18_UNet, self).__init__(n_classes, self.base_channels, level, padding, norm_layer, bilinear)

    def build_encoder(self):
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], base_planes=self.base_channels, b_RGB=self.b_RGB,pretrained=self.pretrained)


class Res18_Unet_MCAtt(nn.Module):
    def __init__(self, n_classes=2, mode="train"):
        super(Res18_Unet_MCAtt, self).__init__()
        self.mode = mode
        self.BackBone = Res18_UNet(n_classes=8, layer=4, b_RGB=True)
        self.seg_head = torch.nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1, stride=1, padding=0), )
        self.attention_head = torch.nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0), )
        self.up2X = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4X = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        if self.mode == "train":
            x_half = F.interpolate(x, scale_factor=0.5)
            x_half = self.BackBone(x_half)
            x = self.BackBone(x)

            x_half_s = self.seg_head(x_half)
            x_s = self.seg_head(x)

            alfa1 = self.attention_head(x_half)
            out = self.up2X(torch.mul(x_half_s, alfa1)) + torch.mul(x_s, (torch.sub(1, self.up2X(alfa1))))

        else:
            x_half = F.interpolate(x, scale_factor=0.5)
            x_2X = F.interpolate(x, scale_factor=2)

            x_half = self.BackBone(x_half)
            x = self.BackBone(x)
            x_2X = self.BackBone(x_2X)

            x_half_s = self.seg_head(x_half)
            x_s = self.seg_head(x)
            x_2X_s = self.seg_head(x_2X)

            x_2X_s = F.interpolate(x_2X_s, scale_factor=0.5)

            alfa2 = self.attention_head(x)
            x_out = torch.mul(x_s, alfa2) + torch.mul(x_2X_s, (torch.sub(1, alfa2)))

            alfa1 = self.attention_head(x_half)
            out = self.up2X(torch.mul(x_half_s, alfa1)) + torch.mul(x_out, torch.sub(1, self.up2X(alfa1)))
        return out


if __name__ == "__main__":

    ipt = torch.rand(1, 3, 128, 768)
    # res18net = Res18_UNet(n_classes=2, layer=4)
    # opt = res18net(ipt)
    # print(opt.shape)
    model = Res18_Unet_MCAtt(n_classes=2, mode="test")
    y = model(ipt)
    print(y.shape)




