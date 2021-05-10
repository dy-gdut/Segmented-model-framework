import torch
import torch.nn as nn
import numpy as np


# 定义双线性插值，作为转置卷积的初始化权重参数
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
       center = factor - 1
    else:
       center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class FCN(nn.Module):
    def __init__(self,num_classes,in_channels=1):
        super(FCN, self).__init__()
        cfg = {
            'VGG16_1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
            'VGG16_2': [512, 512, 512, 'M'],
            'VGG16_3': [512, 512, 512, 'M'],
        }
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.conv_trans1 = nn.Conv2d(512,256,1)
        self.conv_trans2 = nn.Conv2d(256,num_classes,1)

        self.low_features = self._make_layers(in_channels=in_channels, cfg=cfg['VGG16_1'])
        self.median_feature = self._make_layers(in_channels=256, cfg=cfg['VGG16_2'])
        self.high_feature = self._make_layers(in_channels=512, cfg=cfg['VGG16_3'])

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel
        # 2倍上采样
        self.upsample_4x = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(256, 256, 4)  # 使用双线性 kernel
        self.upsample_2x = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        #self.initialize_weights()
        self.upsample_2x.weight.data = bilinear_kernel(512, 512, 4)  # 使用双线性 kernel

        self.out = torch.nn.Softmax()

    def forward(self, x):
        s3 = self.low_features(x)
        # print(s1.shape)
        s4 = self.median_feature(s3)
        # print(s2.shape)
        s5 = self.high_feature(s4)

        scores1 = self.scores1(s5)
        s5 = self.upsample_2x(s5)

        add1 = s4 + s5
        scores2 = self.scores2(add1)
        add1 = self.conv_trans1(add1)
        add1 = self.upsample_4x(add1)
        add2 = add1 + s3

        add2 = self.conv_trans2(add2)

        scores3 = self.upsample_8x(add2)

        return scores3

    def _make_layers(self,in_channels,cfg):
        layers = []
        in_channels = in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def main():
    input = torch.rand([5,1,512,512])
    VGG16 = FCN(num_classes=2)
    output = VGG16(input)
    print(output.shape)


if __name__ == "__main__":
    main()

