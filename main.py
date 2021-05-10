from torch.utils.data import DataLoader
import argparse
from unets.resunet import Res18_UNet
from data_api.data_object_detection_rec import Data_api_rec
from torch import nn
import sys
from utils.visualization import *
from torch.nn import functional as F
from utils.seg_metrics import Seg_metrics
import matplotlib.pyplot as plt
from datetime import datetime
import os
from data_api.get_dataset_mean_std import get_mean_std
from utils.cal_max_area import cal_max_area
import torch
from tqdm import tqdm
from glob import glob
from utils.del_n_directory import del_directory

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)

# 参数初始化 *********
# base_path = os.path.dirname(os.getcwd())
base_path = os.getcwd()

iou_all = []
acc_all = []
loss_all = []
loss_mean = []
best_acc_epoch = 0
best_iou_epoch = 0

# 创建参数管理器 ******
train_parser = argparse.ArgumentParser(description="training---res18Unet on up_facet dataset!")
train_parser.add_argument("--data_root", '-dr', default="/media/root/软件/wqr/data/up_facet2")
train_parser.add_argument("--phase", "-p", default="train", choices=["train", "test", "val"],
                          help="the mode of running")
train_parser.add_argument("--batch_size", "-bs", default=8, type=int)
train_parser.add_argument("--device", "-d", default='cuda:0' if torch.cuda.is_available() else 'cpu')
train_parser.add_argument("--epochs", "-epo", default=100, type=int)
train_parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float)
train_parser.add_argument("--epoch_interval", "-ei", default=10, type=int)
train_parser.add_argument("--val_epoch", "-ve", default=1, type=int)
train_parser.add_argument("--continue_train", "-ct", action="store_true", default=True, help="if continue train using trained model")
train_parser.add_argument("--is_val_best_model", action="store_true", default=True, help="每次最优模型进行验证集的验证，保存结果")
train_parser.add_argument("--state_path", "-sp", default="checkpoints/network_state/network_epo1.pth",
                          help="the path of trained model")

opt = train_parser.parse_args()
print(opt.device)
train_data = Data_api_rec(root=opt.data_root, gen_label=False, updata_txt=False,
                          train_test_ratio=0.8, phase="train", use_relative_path=False)

Norm, [mean, std] = get_mean_std(mean_std_path=os.path.join(base_path, "data_api"), dataset=train_data)

train_data = Data_api_rec(root=opt.data_root, gen_label=False, updata_txt=False,train_test_ratio=0.8, phase="train", use_relative_path=False, Norm=Norm)
val_data = Data_api_rec(root=opt.data_root, gen_label=False, updata_txt=False, train_test_ratio=0.8, phase="val", Norm=Norm)

train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True)
# 为了后续处理方便，val batch_size=1
val_loader = DataLoader(val_data, batch_size=1, shuffle=True)


# model
model = Res18_UNet(n_classes=2, layer=4).to(opt.device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)

# 声明所有可用设备  多GPU
# device_ids = [0, 1]
# device_ids = [0]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])
# loss
criterion = nn.CrossEntropyLoss()
score_criterion = nn.CrossEntropyLoss()

# continue train
start_epoch = 0
if opt.continue_train:
    checkpoint = torch.load(opt.state_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    evaluation_param = checkpoint["evaluation_param"]
    acc_all = evaluation_param['acc']
    iou_all = evaluation_param['iou']
    loss_mean = evaluation_param['loss']
    start_epoch = checkpoint["epoch"]
    print("acc_all:", acc_all)
    print("start train from {} epoch...".format(start_epoch+1))


def main():
    # tensorboard 可视化
    TIMESTAMP = "{0:%Y-%m-%dII%H-%M-%S/}".format(datetime.now())
    log_dir = base_path + '/checkpoints/vis_log/' + TIMESTAMP

    # 文件夹中只保留最新的3个log文件夹,多余删除
    del_directory(base_path + '/checkpoints/vis_log', keep_num=3)
    print("The log save in {}".format(log_dir))
    Vis = VisualBoard(log_dir)

    # 若iou、acc保存在本地，直接用max获取
    # 若是从0开始训练acc_all为[]，直接返回会报错，加上0默认初始化为0（可避免判断acc_all为None再置0）
    best_iou = max(iou_all + [0])
    best_acc = max(acc_all + [0])

    global loss_all
    global model
    for epoch in range(start_epoch+1, opt.epochs):
        model.train()
        batch_nums = int(len(train_data) / opt.batch_size)
        for cnt, (x, y, _, image_label) in enumerate(train_loader):
            x = x.to(opt.device)
            y = y.to(opt.device)

            out = model(x)
            seg_loss = criterion(out, y.squeeze(dim=1).long())
            loss = seg_loss

            # 记录loss
            loss_all.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\repoch:{}/{}-batch:{}/{}-loss:{:.4f} --- best_acc:{}-best_iou:{}'.
                             format(epoch, opt.epochs, cnt, batch_nums, loss, best_acc, best_iou))
            sys.stdout.flush()

        # 计算每一轮的loss
        b_loss = sum(loss_all)/len(loss_all)
        loss_mean.append(b_loss)
        loss_all = []

        # 可视化loss曲线
        Vis.visual_data_curve(name="loss", data=b_loss, data_index=epoch)

        # 验证模式下，关闭梯度回传以及冻结BN层，降低占用内存空间
        with torch.no_grad():
            if epoch % opt.val_epoch == opt.val_epoch - 1:
                model.eval()
                # 验证阶段，每一次返回最优iou以及 最优acc，并保存最优iou、acc的模型参数，同时在tensorboard上可视化recall、acc曲线
                best_iou, best_acc, best_model_flag = validate(best_iou, best_acc, epoch, Vis=Vis, best_model_flag=False)
                # 可视化训练集的训练效果
                for cnt, (x, y, _, image_label) in enumerate(train_loader):
                    seg_vis = model(x.to(opt.device))
                    seg_vis = F.softmax(seg_vis, dim=1)
                    seg_vis = seg_vis[0]
                    # tensorboard 可视化4张图片
                    if cnt < 4:
                        # 上采样到原图尺寸
                        seg_pil = trans(seg_vis[1].cpu())
                        seg_vis = transf(seg_pil)
                        # 由于数据做了归一化，可视化要做反归一化
                        mean_values = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
                        std_values = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
                        x = x * std_values + mean_values
                        vis_image = torch.cat((seg_vis, x.narrow(1, 1, 1)[0], y.narrow(1, 0, 1)[0]), dim=1)
                        # 可视化训练效果，每一次验证可视化4张，[分割图，原图，标签图]
                        Vis.visual_image(vis_image, data_index=epoch, tag="epoch{}_out".format(epoch))
                    if cnt == 4:
                        break

                # 验证每一次最优模型，保存验证结果
                if best_model_flag & opt.is_val_best_model:
                    for val_image, val_label, image_path, _ in tqdm(val_loader):
                        seg_val = model(val_image.to(opt.device))
                        seg_val = F.softmax(seg_val, dim=1)
                        seg_val = seg_val[0][1]

                        # 由于数据做了归一化，可视化要做反归一化
                        mean_values = torch.tensor(mean, dtype=val_image.dtype).view(3, 1, 1)
                        std_values = torch.tensor(std, dtype=val_image.dtype).view(3, 1, 1)
                        val_image = val_image * std_values + mean_values
                        val_image = val_image.squeeze(dim=0).permute(1, 2, 0)
                        val_label = val_label[0].squeeze(dim=0)
                        val_label = np.uint8(val_label * 255)

                        # 可视化图像
                        seg_val = seg_val.detach().cpu()
                        seg_val = np.uint8(seg_val * 255)
                        _, out_cv = cv2.threshold(seg_val, 128, 255, cv2.THRESH_BINARY)
                        max_area = cal_max_area(out_cv)

                        concat_image = Concat3CImage([np.uint8(val_image * 255), val_label, seg_val],
                                                     mode='Col', offset=1)

                        val_save_path = "checkpoints/val_image/epoch{}/{}".format(epoch, image_path[0].split("/")[-1])
                        if not os.path.exists("checkpoints/val_image/epoch{}".format(epoch)):
                            os.mkdir("checkpoints/val_image/epoch{}".format(epoch))
                        cv2.imwrite(val_save_path, concat_image)

        if epoch % opt.epoch_interval == opt.epoch_interval - 1:
            network_state = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'epoch': epoch,
                             'evaluation_param': {'acc': acc_all, 'iou': iou_all, 'loss': loss_mean}
                             }

            torch.save(network_state, base_path + '/checkpoints/network_state/network_epo{}.pth'.format(epoch))

            print('\nsave network_epo{}.pth in checkpoints/network_state successfully!'.format(epoch))

    Vis.visual_close()


# 保存每一次验证的iou、loss、acc最终绘制曲线图
def dis_plt():
    global base_path
    global loss_mean
    plt.figure()
    plt.plot(np.arange(0, len(acc_all), 1)*opt.val_epoch, acc_all, marker='*', color='b', label='acc')
    plt.legend()
    # 横坐标名称
    plt.xlabel('epoch')
    # 纵坐标名称
    plt.ylabel('val_acc')
    plt.savefig(base_path + '/checkpoints/visual_result/acc_all.png')

    plt.cla()
    plt.plot(np.arange(0, len(iou_all), 1)*opt.val_epoch, iou_all, marker='o', color='r', label='iou')
    plt.legend()
    # 横坐标名称
    plt.xlabel('epoch')
    # 纵坐标名称
    plt.ylabel('val_m_iou')
    plt.savefig(base_path + '/checkpoints/visual_result/iou_all.png')

    plt.cla()
    plt.plot(np.arange(0, len(loss_mean), 1), loss_mean, marker='*', color='b', label='loss')
    plt.legend()
    # 横坐标名称
    plt.xlabel('epoch')
    # 纵坐标名称
    plt.ylabel('loss')
    plt.savefig(base_path + '/checkpoints/visual_result/loss_all.png')


def create_directory():
    root = ["checkpoints/visual_result",
            "checkpoints/vis_log",
            "checkpoints/test_result",
            "checkpoints/network_state",
            "checkpoints/val_image"]
    for path in root:
        if not os.path.exists(path):
            os.mkdir(path)


def validate(best_iou, best_acc, epoch, Vis=None, best_model_flag=False):
    acc_metrics = Seg_metrics(num_classes=2)
    iou_metrics = Seg_metrics(num_classes=2)
    global best_acc_epoch
    global best_iou_epoch
    global base_path
    global model
    model.eval()
    for cnt, (x, y, _, image_label) in enumerate(val_loader):
        output = model(x.to(opt.device))
        output = F.softmax(output, dim=1)
        output = output.squeeze(dim=0)
        out = output[1]

        out_image = trans(out.cpu())  # cpu
        label_image = trans(y[0])
        out_image = np.where(np.array(out_image) > 128, 1, 0)
        label_image = np.where(np.array(label_image) > 254, 1, 0)

        out_cv1 = out.detach().cpu()
        out_cv1 = np.uint8(out_cv1 * 255)
        _, out_cv = cv2.threshold(out_cv1, 128, 255, cv2.THRESH_BINARY)
        max_area = cal_max_area(out_cv)

        # 只根据阈值对分割图进行分类
        # y1 = np.max(np.array(out_cv))
        # 在阈值分割的基础上根据连通区域的大小进行分类
        y1 = 1 if max_area > 0 else 0
        # y1 = np.max(out_image)
        if y1 == 1:
            y1 = np.array([1])
        else:
            y1 = np.array([0])
        if image_label == 1:
            label = np.array([1])
        else:
            label = np.array([0])
        acc_metrics.add_batch(label, y1)

        # cal mean_iou
        iou_metrics.add_batch(label_image.reshape(1, -1), out_image.reshape(1, -1))

    acc = acc_metrics.pixelAccuracy()
    recall = acc_metrics.TPR()

    cur_acc = round(acc * 100, 2)
    acc_all.append(cur_acc)

    iou = iou_metrics.meanIntersectionOverUnion()
    cur_iou = round(iou * 100, 2)
    iou_all.append(cur_iou)

    if cur_iou > best_iou:
        best_iou = cur_iou
        best_iou_epoch = epoch
        torch.save(model.state_dict(), 'checkpoints/network_state/best_iou_model.pth')
        print('\nsave best_iou_model.pth successfully in the {} epoch!'.format(epoch))

    if cur_acc > best_acc:
        best_model_flag = True
        best_acc = cur_acc
        best_acc_epoch = epoch

        # 避免多次保存相同epoch的pth文件
        remove_old_pths = glob("checkpoints/network_state/epoch{}*".format(epoch))
        for remove_old_pth in remove_old_pths:
            if os.path.exists(remove_old_pth):
                os.remove(remove_old_pth)

        torch.save(model.state_dict(), 'checkpoints/network_state/epoch{}_acc{}_model.pth'.format(epoch, best_acc))

        print('\nsave best_acc_model.pth successfully in the {} epoch!'.format(epoch))

    text_note_iou = "The best_iou gens in the {}_epoch, the best iou is {}". \
        format(best_iou_epoch, best_iou)
    text_note_acc = "The best_acc gens in the {}_epoch,the best acc is {}". \
        format(best_acc_epoch, best_acc)
    text_note_recall = "the recall is {}".format(round(recall, 2))

    # 最优acc、iou保存路径提示
    Vis.writer.add_text(tag="note", text_string=text_note_iou + "||" + text_note_acc + "," + text_note_recall,
                        global_step=epoch)
    Vis.visual_data_curve(name="acc", data=cur_acc, data_index=epoch)
    Vis.visual_data_curve(name="iou", data=cur_iou, data_index=epoch)
    print("\nepoch:{}-val_acc:{}--val_iou:{}".format(epoch, cur_acc, cur_iou))
    return best_iou, best_acc, best_model_flag


if __name__ == '__main__':
    # 避免文件夹不存在运行错误
    create_directory()
    main()
    dis_plt()

