# 画第4章篡改区域预测结果
import torch
import cv2
import utils
import numpy as np
import collections
import resnet34_cons_emb
import nedbnet
from torch.nn import functional as F

from matplotlib import pyplot as plt
from matplotlib import rcParams
# rcParams['font.family'] = 'SimHei'
from matplotlib.font_manager import FontProperties

import random

# todo 记得删除，包括服务器项目下边的4和5文件夹
# 画第3章最后的篡改区域预测
if __name__ == '__main__':
  for i in range(10):
    name = random.randint(1, 80000)
    name = str(name).zfill(5)
    print(name)

    # name = "04458"

    # 0. 加载图像
    image = cv2.imread("/home/kxg/zzz/datasets/coco_stuff/coco_tampers/Tp/" + name + ".jpg")
    mask = cv2.imread("/home/kxg/zzz/datasets/coco_stuff/coco_tampers/Gt/" + name + ".png", 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)

    # name = "10830"  # 用的没加cons_conv的最后的模型结果
    # image = cv2.imread("D:/tamperature_datas/coco_tampers/Tp/" + name + ".jpg")
    # mask = cv2.imread("D:/tamperature_datas/coco_tampers/Gt/" + name + ".png", 0)

    # name = "10831"  # 用的没加cons_conv的最后的模型结果
    # image = cv2.imread("D:/tamperature_datas/coco_tampers/Tp/" + name + ".jpg")
    # mask = cv2.imread("D:/tamperature_datas/coco_tampers/Gt/" + name + ".png", 0)

    mask_gt = np.where(mask > 128, 1, 0)
    image_resize = cv2.resize(image, (512, 512))
    image_resize = utils.input_transform(image_resize, [0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
    image_resize = image_resize.transpose((2, 0, 1))
    image_resize = torch.Tensor(image_resize).unsqueeze(0).cuda()

    # cv2.imshow("image", image)
    # cv2.imshow("mask", mask)

    # 1.1 ResNet34+噪声+边缘预测
    # 加载模型权重
    model = resnet34_cons_emb.get_seg_model(None).eval()
    model = model.cuda()
    pretrained_dict = torch.load("./4/resnet34_emb.pth")
    model_dict = model.state_dict()

    filtered_dict = collections.OrderedDict()
    for k, v in pretrained_dict.items():
      filtered_dict[k[7:]] = v
    pretrained_dict = filtered_dict

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 得到模型输出；如果模型有多个输出结果，选择配置文件设定的最终输出结果；将结果恢复到图像原始大小；
    mask_pred, edge_pred = model(image_resize)
    mask_pred = F.interpolate(input=mask_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=False)
    mask_pred = torch.sigmoid(mask_pred)
    # mask_pred = (mask_pred.squeeze(0).squeeze(0) >= 0.5).long().numpy()
    resnet34_cons_emb_pred = ((mask_pred.squeeze(0).squeeze(0)).detach().cpu().numpy() * 255).astype(np.uint8)

    edge_pred = F.interpolate(input=edge_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=False)
    edge_pred = torch.sigmoid(edge_pred)
    resnet34_cons_emb_edge_pred = ((edge_pred.squeeze(0).squeeze(0)).detach().cpu().numpy() * 255).astype(np.uint8)

    # cv2.imshow("resnet34_cons_imp_pred", resnet34_cons_imp_pred)

    # 1.2 双分支+噪声+边缘
    model = nedbnet.get_seg_model(None).eval()
    model = model.cuda()
    pretrained_dict = torch.load("./5/DB.pth")
    model_dict = model.state_dict()

    filtered_dict = collections.OrderedDict()
    for k, v in pretrained_dict.items():
      filtered_dict[k[7:]] = v
    pretrained_dict = filtered_dict

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 得到模型输出；如果模型有多个输出结果，选择配置文件设定的最终输出结果；将结果恢复到图像原始大小；
    mask_pred, edge_pred = model(image_resize)
    mask_pred = F.interpolate(input=mask_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=False)
    mask_pred = torch.sigmoid(mask_pred)
    # mask_pred = (mask_pred.squeeze(0).squeeze(0) >= 0.5).long().numpy()
    db_pred = ((mask_pred.squeeze(0).squeeze(0)).cpu().detach().numpy() * 255).astype(np.uint8)

    edge_pred = F.interpolate(input=edge_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=False)
    edge_pred = torch.sigmoid(edge_pred)
    db_edge_pred = ((edge_pred.squeeze(0).squeeze(0)).detach().cpu().numpy() * 255).astype(np.uint8)
    # cv2.imshow("resnet34_cons_imp_pred", resnet34_cons_imp_pred)



    # 1.4 双分支+噪声+边缘+NL-D
    model = nedbnet.get_seg_model(None).eval()
    model = model.cuda()
    pretrained_dict = torch.load("./5/DB-NL-D.pth")
    model_dict = model.state_dict()

    filtered_dict = collections.OrderedDict()
    for k, v in pretrained_dict.items():
      filtered_dict[k[7:]] = v
    pretrained_dict = filtered_dict

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # 得到模型输出；如果模型有多个输出结果，选择配置文件设定的最终输出结果；将结果恢复到图像原始大小；
    mask_pred, edge_pred = model(image_resize)
    mask_pred = F.interpolate(input=mask_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=False)
    mask_pred = torch.sigmoid(mask_pred)
    # mask_pred = (mask_pred.squeeze(0).squeeze(0) >= 0.5).long().numpy()
    db_nl_d_pred = ((mask_pred.squeeze(0).squeeze(0)).cpu().detach().numpy() * 255).astype(np.uint8)

    edge_pred = F.interpolate(input=edge_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=False)
    edge_pred = torch.sigmoid(edge_pred)
    db_nl_d_edge_pred = ((edge_pred.squeeze(0).squeeze(0)).detach().cpu().numpy() * 255).astype(np.uint8)
    # cv2.imshow("resnet34_cons_imp_pred", resnet34_cons_imp_pred)

    # cv2.waitKey(0)

    # 画图
    plt.figure(figsize=(24, 5))
    font = FontProperties(fname="/home/kxg/zzz/SimHei.ttf", size=30)

    # 1. 原始篡改图像
    plt.subplot(1, 6, 1)
    plt.axis('off')
    plt.title("篡改图像", fontproperties=font)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 2. 篡改图像对应的mask
    plt.subplot(1, 6, 2)
    plt.axis('off')
    plt.title("篡改图像的mask", fontproperties=font)
    plt.imshow(mask, cmap='gray')

    # 3. 边缘
    plt.subplot(1, 6, 3)
    plt.axis('off')
    plt.title("篡改边缘的mask", fontproperties=font)
    plt.imshow(edge, cmap='gray')

    # 3. ResNet+噪声+边缘
    plt.subplot(1, 6, 4)
    plt.axis('off')
    plt.title("ResNet+噪声+边缘", fontproperties=font)
    plt.imshow(resnet34_cons_emb_edge_pred, cmap='gray')

    # 4. 双分支+噪声+边缘
    plt.subplot(1, 6, 5)
    plt.axis('off')
    plt.title("双分支+噪声+边缘", fontproperties=font)
    plt.imshow(db_edge_pred, cmap='gray')

    # 6. 双分支+噪声+边缘+NL-D
    plt.subplot(1, 6, 6)
    plt.axis('off')
    plt.title("双分支+噪声\n+边缘+NL-D", fontproperties=font)
    plt.imshow(db_nl_d_edge_pred, cmap='gray')
    # plt.imshow(resnet34_emb_pred, cmap='gray')

    # 自当调整子图的分布
    plt.tight_layout()
    plt.show()



