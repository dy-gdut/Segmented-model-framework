import argparse
from unets.resunet import Res18_UNet
import torch
from tqdm import tqdm
from utils.seg_metrics import Seg_metrics
import numpy as np
from PIL import Image
from data_api.get_dataset_mean_std import get_mean_std
import os
import shutil
import cv2
from glob import glob
from torchvision import transforms as T
from torch.nn import functional as F
from utils.visualization import Concat3CImage
from utils.cal_max_area import cal_max_area

base_path = os.getcwd()
test_parser = argparse.ArgumentParser(description="testing---res18unet_attention_decision on up_facet dataset!")
test_parser.add_argument("--data_root", '-dr', default="/media/root/软件/wqr/data/up_facet2")
test_parser.add_argument("--phase", "-p", default="test", choices=["train", "test", "val"],
                         help="the mode of running")
test_parser.add_argument("--device", "-d", default='cuda:0' if torch.cuda.is_available() else 'cpu')
test_parser.add_argument("--batch_size", "-bs", default=1, type=int)
test_parser.add_argument("--continue_train", "-ct", default=False, type=bool,
                         help="continue train using trained model")
test_parser.add_argument("--state_path", "-sp", default="checkpoints/network_state/acc94.74_model.pth",
                         help="the path of trained model")
opt = test_parser.parse_args()
print(opt.device)
# 获取数据集的均值、方差
Norm, [mean, std] = get_mean_std(mean_std_path=os.path.join(base_path, "data_api"))
print(mean)
print(std)
# model
model = Res18_UNet(n_classes=2, layer=4).to(opt.device)
model.eval()
checkpoints = torch.load(os.path.join(base_path, opt.state_path))
# model.load_state_dict(checkpoints["model"])
model.load_state_dict(checkpoints)


metrics = Seg_metrics(num_classes=2)

# 清空文件夹
shutil.rmtree(base_path + '/checkpoints/test_single_result')
os.mkdir(base_path + '/checkpoints/test_single_result')

# 遍历的文件夹
image_paths = glob("/media/root/软件/wqr/data/test_single/*.bmp")
trans = T.Compose([T.Resize([128, 768]),
                   T.ToTensor()])
trans_pil = T.ToPILImage()

# 遍历文件夾對圖片進行分类，不用数据loader
for i, image_path in enumerate(tqdm(image_paths)):
    img = cv2.imread(image_path)
    image_array = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil = Image.fromarray(image_array)
    # 数据处理
    img_tensor = trans(img_pil)
    img_tensor = Norm(img_tensor)
    img_tensor = img_tensor.unsqueeze(dim=0).to(opt.device)
    output = model(img_tensor)
    # 模型输出的数据转换
    output = F.softmax(output, dim=1)
    out = output.squeeze(dim=0)[1]
    out = out.detach().cpu()
    out_cv1 = np.uint8(out*255)
    _, out_cv = cv2.threshold(out_cv1, 80, 255, cv2.THRESH_BINARY)
    cv2.imshow("a", out_cv)
    cv2.waitKey()
    max_area = cal_max_area(out_cv)
    print(max_area)
    # 只根据阈值对分割图进行分类
    # y = np.max(np.array(out_cv))
    # 在阈值分割的基础上根据连通区域的大小进行分类
    y = 1 if max_area > 200 else 0

    if not os.path.exists(base_path + '/checkpoints/test_single_result/P/'):
        os.makedirs(base_path + '/checkpoints/test_single_result/P/')
    if not os.path.exists(base_path + '/checkpoints/test_single_result/F/'):
        os.makedirs(base_path + '/checkpoints/test_single_result/F/')

    # 根据预测分类
    if y == 0:
        img = cv2.resize(img, (768, 128))
        concat_img = Concat3CImage([img, out_cv], mode="Col")
        cv2.imwrite(base_path + '/checkpoints/test_single_result/P/' + image_path.split("/")[-1], concat_img)

    else:
        img = cv2.resize(img, (768, 128))
        concat_img = Concat3CImage([img, out_cv], mode="Col")
        cv2.imwrite(base_path + '/checkpoints/test_single_result/F/' + image_path.split("/")[-1], concat_img)







