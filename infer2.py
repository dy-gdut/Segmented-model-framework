import argparse
from torch.utils.data import DataLoader
from unets.resunet import Res18_UNet
import torch
from torchvision import transforms as tran
import cv2
from utils.visualization import Concat3CImage
from tqdm import tqdm
from torch.nn import functional as F
from utils.seg_metrics import Seg_metrics
import numpy as np
from data_api.data_image_label import Data_image_label
from data_api.get_dataset_mean_std import get_mean_std
import os
import shutil
from utils.cal_max_area import cal_max_area
from glob import glob

# base_path = os.path.dirname(os.getcwd())
base_path = os.getcwd()

test_parser = argparse.ArgumentParser(description="testing---res18unet_attention_decision on up_facet dataset!")
test_parser.add_argument("--data_root", '-dr', default="/media/root/软件/wqr/data/up_facet_2021.1")
test_parser.add_argument("--phase", "-p", default="test", choices=["train", "test", "val"],
                         help="the mode of running")
test_parser.add_argument("--device", "-d", default='cuda:0' if torch.cuda.is_available() else 'cpu')
test_parser.add_argument("--batch_size", "-bs", default=1, type=int)
test_parser.add_argument("--continue_train", "-ct", default=False, type=bool,
                         help="continue train using trained model")
test_parser.add_argument("--state_path", "-sp", default="checkpoints/network_state/acc94.74_model.pth",
                         help="the p500ath of trained model")

opt = test_parser.parse_args()
print(opt.device)
# 获取数据集的均值、方差
Norm, [mean, std] = get_mean_std(mean_std_path=os.path.join(base_path, "data_api"))
test_data = Data_image_label(root=opt.data_root, updata_txt=True,
                             train_test_ratio=0.0, phase="test", Norm=Norm)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

# model
model = Res18_UNet(n_classes=2, layer=4).to(opt.device)

# device_ids = [0]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])

model.eval()
checkpoints = torch.load(os.path.join(base_path, opt.state_path))
# model.load_state_dict(checkpoints["model"])
model.load_state_dict(checkpoints)

trans = tran.ToPILImage()
metrics = Seg_metrics(num_classes=2)
result_acc = []
result_iou = []

c_name = ["TP", "FN", "FP", "TN"]
# 清空文件夹
shutil.rmtree(base_path + '/checkpoints/test_result')
os.mkdir(base_path + '/checkpoints/test_result')

for i, (x, y, image_path, image_label) in enumerate(tqdm(test_loader)):
    output = model(x.to(opt.device))
    output = F.softmax(output, dim=1)
    output = output.squeeze(dim=0)
    out = output[1].unsqueeze(0)

    # 反归一化
    mean_values = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_values = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    x = x * std_values + mean_values
    x = x.squeeze(dim=0)
    x = x.permute(1, 2, 0)
    y = y.squeeze(dim=0).squeeze(dim=0)
    y = np.uint8(y*255)
    # 可视化图像
    out_cv1 = out.detach().cpu().squeeze(dim=0)
    out_cv1 = np.uint8(out_cv1 * 255)
    _, out_cv = cv2.threshold(out_cv1, 80, 255, cv2.THRESH_BINARY)
    concat_image = Concat3CImage([np.uint8(x * 255), out_cv], mode='Col', offset=1)

    max_area = cal_max_area(out_cv)
    # 只根据阈值对分割图进行分类
    # y1 = np.max(np.array(out_cv))
    # 在阈值分割的基础上根据连通区域的大小进行分类
    y1 = 1 if max_area > 200 else 0

    # 利用混淆矩阵计算acc
    # y1 = np.max(out_cv)
    if y1 == 1:
        y1 = np.array([1])
    else:
        y1 = np.array([0])
    if image_label == 1:
        label = np.array([1])
    else:
        label = np.array([0])

    metrics.add_batch(label, y1)
    acc = metrics.pixelAccuracy()
    # 根据混淆矩阵对分类结果进行保存，对应TP、 FP、 FN、 TN
    confusionMatrix = metrics.confusionMatrix
    metrics.reset()
    confusionMatrix = confusionMatrix.reshape(1, -1)
    image_save_path = base_path + '/checkpoints/test_result/{}'.format(c_name[np.argmax(confusionMatrix, axis=1)[0]])
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    image_save_path = os.path.join(image_save_path, image_path[0].split("/")[-1])
    cv2.imwrite(image_save_path, concat_image)
    result_acc.append(acc)

result = 0
for acc in result_acc:
    result += acc
print("acc:{}".format(round(result * 100 / len(result_acc), 2)))


TP = len(glob(base_path + '/checkpoints/test_result/TP/*'))
TN = len(glob(base_path + '/checkpoints/test_result/TN/*'))
FP = len(glob(base_path + '/checkpoints/test_result/FP/*'))
FN = len(glob(base_path + '/checkpoints/test_result/FN/*'))
# 真阳性率,漏检率
TPR = round(FN/(TP+FN), 2)
# 假阳性率，误检率
FPR = round(FP/(FP+TN), 2)
print("漏检率：{}, 误检率: {}".format(TPR, FPR))



