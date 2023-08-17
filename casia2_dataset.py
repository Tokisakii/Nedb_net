import os
import cv2
import numpy as np
import torch
import torch.utils.data
import utils
from config import Config
import random


# 数据集的基类，构建图像和对应的gt
class Casia2Dataset(torch.utils.data.Dataset):
  # 1. root和list_path分别未数据集的目录以及要训练数据图片的具体列表（train.txt）
  def __init__(self, root, list_path):
    self.root = root
    self.list_path = list_path

    # 从保存训练或者测试使用的图像名字文件中，得到所有的图像路径信息
    self.img_list = [line.strip() for line in open(root + list_path)]

    # 从图像路径数组中，遍历每一条路径，根据路径得到对应的文件的路径、label路径、文件名
    self.files = self.read_files()

    # 类别的损失占的权重
    self.class_weights = torch.FloatTensor([1, 1]).cuda()

  # 2. 从图像路径数组中，遍历每一条路径，根据路径得到对应的文件的路径、label路径、文件名、权重
  def read_files(self):
    files = []
    for item in self.img_list:
      name = os.path.splitext(item)[0]  # 将 123.txt 变成 123
      image_path = os.path.join(self.root, 'Tp', item)
      label_path = os.path.join(self.root, 'Gt', name + '_gt.png')
      files.append({
        "img": image_path,
        "label": label_path,
        "name": name
      })
    return files

  # 3. 返回了训练集/测试集图像数量
  def __len__(self):
    return len(self.files)

  # 4. 拿数据的方法
  def __getitem__(self, index):
    # 0. 根据index从files里拿到该index对应图像的信息，并读入图像
    item = self.files[index]
    name = item["name"]
    image = cv2.imread(item["img"], cv2.IMREAD_COLOR)

    # 1. 将图像缩放到 512 x 512
    image = cv2.resize(image, (Config.base_size, Config.base_size))

    # 2. 将对应的gt也读取进来，并且也缩放到 512 x 512
    mask_origin = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask_origin, (int(Config.base_size), int(Config.base_size)))


    # 4. 镜像
    if Config.aug:
      # 1. 3种翻转
      flip = random.randint(0, 3)
      if flip == 1:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
      elif flip == 2:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
      elif flip == 3:
        image = cv2.flip(image, -1)
        mask = cv2.flip(mask, -1)

      # 2. 3种旋转
      rotate = random.randint(0, 3)
      if rotate == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
      elif rotate == 2:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
      elif rotate == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
        mask = cv2.rotate(mask, cv2.ROTATE_180)

      # 3. 模糊 or 锐化，这里故意让0和4作为不处理，增大原图几率
      # tmp = random.randint(0, 3)
      # if tmp == 1:
      #   image = cv2.GaussianBlur(image, (5, 5), 0)
      # elif tmp == 2:
      #   sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
      #   image = cv2.filter2D(image, cv2.CV_32F, sharpen_op)

      # 4. cm or sp，这里故意让0和3作为不处理，增大原图几率
      # tmp = random.randint(0, 3)
      # if tmp == 1 or tmp == 2:
      #   y_dst = random.randint(10, 480)
      #   x_dst = random.randint(10, 480)
      #   y_src = random.randint(10, 480)
      #   x_src = random.randint(10, 480)
      #
      #   h = random.randint(20, 128)
      #   w = random.randint(20, 128)
      #
      #   if y_dst + h > 500:
      #     h = 500 - y_dst
      #   if x_dst + w > 500:
      #     w = 500 - x_dst
      #
      #   if y_src + h > 500:
      #     h = 500 - y_src
      #   if x_src + w > 500:
      #     w = 500 - x_src
      #
      #   if tmp == 1:
      #     source = image
      #   else:
      #     path = item["img"].split("Tp")[0] + "Au/"
      #     names = os.listdir(path)
      #     names = [name for name in names if name.endswith(".jpg")]
      #     index = random.randint(0, len(names) - 1)
      #     source_image_path = path + names[index]
      #     source = cv2.imread(source_image_path)
      #     source = cv2.resize(source, (Config.base_size, Config.base_size))
      #
      #   image[y_dst+h, x_dst+w, :] = source[y_src+h, x_src+w, :]
      #   mask[y_dst+h, x_dst+w] = 255

    # 5. 生成篡改边缘
    edge = utils.make_edge(cv2.resize(mask, (int(Config.base_size / 4), int(Config.base_size / 4))))
    mask = np.where(mask >= 127, 1, 0)
    edge = np.where(edge >= 127, 1, 0)

    # 6. 图像归一化
    image = utils.input_transform(image, Config.mean, Config.std)
    image = image.transpose((2, 0, 1)).astype(np.float32)
    # mask = mask.astype(np.float32)
    # edge = edge.astype(np.float32)

    return image.copy(), mask.copy(), edge.copy(), name
