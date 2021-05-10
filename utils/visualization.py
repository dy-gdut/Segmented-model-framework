from PIL import Image, ImageDraw
import numpy as np
import cv2
from torchvision import transforms
from tensorboardX import SummaryWriter
trans = transforms.ToPILImage()
transf = transforms.ToTensor()


def gen_result(data, gen_image, fill_color=0):
    image_save_list = []
    cols = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(3, 3))
    for index in range(min(gen_image.shape[0], cols)):
        if isinstance(data, list):
            img = cv2.cvtColor(np.uint8(tensor_to_PIL(data[1][index])), cv2.COLOR_GRAY2BGR)
            gen_img = cv2.cvtColor(np.uint8(tensor_to_PIL(gen_image[index])), cv2.COLOR_GRAY2BGR)
            sub_img = cv2.subtract(gen_img, img)
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            sub_img = np.where(sub_img > 8, 255, 0)
            images = [tensor_to_PIL(data[0][index]), tensor_to_PIL(data[2][index]),
                      tensor_to_PIL(data[1][index]), tensor_to_PIL(gen_image[index]), sub_img]
            image_save_block = ConcatImage(images, text=["src", "mask", "defect_img", "gen_img", "sub_img"],
                                           offset=5, color=fill_color)
        else:
            img = cv2.cvtColor(np.uint8(tensor_to_PIL(data[index])), cv2.COLOR_GRAY2BGR)
            gen_img = cv2.cvtColor(np.uint8(tensor_to_PIL(gen_image[index])), cv2.COLOR_GRAY2BGR)
            sub_img = cv2.subtract(gen_img, img)
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            # sub_img = cv2.threshold(sub_img, 8, 255, cv2.THRESH_BINARY)
            sub_img = np.uint8(np.where(sub_img > 8, 255, 0))
            sub_img = cv2.morphologyEx(sub_img, cv2.MORPH_OPEN, kernel=kernel)
            images = [tensor_to_PIL(data[index]), tensor_to_PIL(gen_image[index]), sub_img]

            image_save_block = ConcatImage(images, text=["src", "gen_img", "sub_img"],
                                           offset=5, color=fill_color)
        image_save_list.append(image_save_block)
    image = ConcatImage(image_save_list, mode="Col", offset=5, color=fill_color)
    return image


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(dim=0)
    tran = transforms.ToPILImage()
    return tran(image)


def pic_text(image, text):
    # 新建绘图对象
    draw = ImageDraw.Draw(image)
    # 根据字符串长度确定底矩形长
    length = 1
    for _ in text:
        length += 1
    draw.rectangle([(0, 0), (5*length, 10)], fill='#000000')
    draw.text((0, 0), text, font=None, fill="#ffffff", direction=None)
    return image


# text = ["src", "label", "output", "crf_image"]
# image_list = [src_image, label_image, out_image, crf_image]
# concat_image = ConcatImage(images=image_list, text=text, mode='Col', offset=1)
def ConcatImage(images, text=None, mode="Adapt", scale=1.0, offset=None, color=0):
    """
    :param images: 图片列表
    :param mode:   图片排列方式["Row" ,"Col","Adapt"]
    :param scale:  图片缩放比例
    :param offset: 图片间距
    :return:
    """
    if not isinstance(images, list):
        raise Exception('images must be a  list')
    # text为none则不显示，不为none要求字符串列表
    if text is not None:
        print_text = True
        if not isinstance(text, list):
            raise Exception('text must be a  list')
        if len(images) != len(text):
            raise Exception('The length of text and images must be same!')
    else:
        print_text = False
    if mode not in ["Row", "Col", "Adapt"]:
        raise Exception('mode must be "Row" ,"Adapt",or "Col"')

    images = [np.uint8(img) for img in images]   # if Gray  [H,W] else if RGB  [H,W,3]
    images = [img.squeeze(2) if len(img.shape) > 2 and img.shape[2] == 1 else img for img in images]
    count = len(images)
    img_ex = Image.fromarray(images[0])
    size = img_ex.size   # [W,H]
    size = [int(size[0] * scale), int(size[1] * scale)]

    if mode == "Adapt":
        mode = "Row" if size[0] <= size[1] else "Col"
    if offset is None:
        offset = int(np.floor(size[0] * 0.02))
    if mode == "Row":
        target = Image.new(img_ex.mode, (size[0] * count+offset*(count-1), size[1] * 1), color=color)
        for i in range(count):
            image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
            if print_text:
                image = pic_text(image, text=text[i])
            target.paste(image, (i*(size[0]+offset), 0))
            # target.paste(image, (i * (size[0] + offset), 0, i * (size[0] + offset) + size[0], size[1]))
        return target
    if mode == "Col":
        target = Image.new(img_ex.mode, (size[0], size[1] * count+offset*(count-1)), color=color)
        for i in range(count):
            image = Image.fromarray(images[i]).resize(size, Image.BILINEAR).convert(img_ex.mode)
            if print_text:
                image = pic_text(image, text=text[i])
            target.paste(image, (0, i*(size[1]+offset)))
        return target


# accept cv_image
def Concat3CImage(image_list, mode="Adapt", offset=None, fill_color=(0,0,0), scale=1.0):
    if not isinstance(image_list, list):
        raise Exception('images must be a  list  ')
    if mode not in ["Row", "Col"]:
        raise Exception('mode must be "Row" ，“Adapt" or "Col"')
    size = image_list[0].shape[:2]
    images = []
    for img in image_list:
        if img.shape == size:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)

    images = [np.uint8(img) for img in images]
    count = len(images)
    if offset is None:
        offset = int(np.floor(size[0] * 0.02))

    if mode == "Adapt":
        mode = "Row" if size[0] <= size[1] else "Col"

    model_row = np.ones((size[0] * 1, size[1] * count+offset*(count-1), 3))
    model_col = np.ones((size[0] * count + offset * (count - 1), size[1] * 1, 3))

    if mode == "Row":
        target = np.ones_like(model_row)
        for channel in range(3):
            target[:,:,channel] = fill_color[channel]
        for i in range(count):
            target[0:(size[0]), i*size[1]+offset*i:(i+1)*size[1]+offset*i, :] = images[i]
    elif mode == "Col":
        target = np.ones_like(model_col)
        for channel in range(3):
            target[:,:,channel] = fill_color[channel]
        for i in range(count):
            target[i*size[0]+offset*i:(i+1)*size[0]+offset*i, 0:(size[1]), :] = images[i]
    else:
        target = np.ones_like(model_row) * fill_color

    target = np.uint8(target)
    output = cv2.resize(target,dsize=(int(target.shape[1]*scale), int(target.shape[0]*scale)))
    return output


# 可在浏览器中实时查看数据模型
# 在cmd中输入tensorboard --logdir 'log' 输出 查看地址

class VisualBoard(object):
    def __init__(self, save_dir):
        self.writer = SummaryWriter(save_dir, comment="log")

    def visual_model(self, model, x_input):
        self.writer.add_graph(model, (x_input,))

    def visual_data_curve(self, name=None, data=None, data_index=None):
        if name is None:
            name='data'
        if data is None or data_index is None:
            print("Please input data or data_index!!")
            return None
        self.writer.add_scalar(name,data,global_step=data_index)

    def visual_data_curves(self, name='data', data_dict=None, data_index=None):
        if data_dict is None or data_index is None:
            print("Please input data or data_index!!")
            return None
        assert isinstance(data_dict,dict)
        self.writer.add_scalars(name, data_dict, data_index)

    def visual_image(self, img, data_index=0, tag=None):
        """
        :param img: tensor类型数据
        :param data_index:
        :return:
        """
        self.writer.add_image(tag=tag, img_tensor=img, global_step=data_index)

    def visual_close(self):
        self.writer.close()


def main():
    Vis = VisualBoard('log')
    img_dir = 'D:/qr_project/base_project/1.jpg'
    save_dir = 'D:/qr_project/base_project/2.jpg'
    img = Image.open(img_dir)
    img_list = [img, img, img]
    target = ConcatImage(img_list, mode="Col", offset=5, scale=0.5)
    target.save(save_dir)
    tran = transforms.ToTensor()
    img_tensor = tran(target)
    Vis.visual_image(img_tensor)
    print(target.size)
    Vis.visual_close()


if __name__ == '__main__':
    main()
