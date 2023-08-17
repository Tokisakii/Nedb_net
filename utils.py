import cv2
import math
import torch
from numba import jit


# 1. 简单的日志打印函数
def log(text, array=None):
  if array is not None:
    text = text.ljust(25)
    text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
      str(array.shape),
      array.min() if array.size else "",
      array.max() if array.size else ""))
  print(text)


# 2. 输入归一化
def input_transform(image, mean, std):
  image = image / 255.0
  image -= mean
  image /= std
  return image


# 3. 将单张图转成tensor
def image_to_tensor(image, mean, std):
  image = input_transform(image, mean, std)
  image = image.transpose((2, 0, 1))
  image = torch.Tensor(image)
  return image.unsqueeze(0).cuda()


# 5. 根据mask，获得对应的edge
def make_edge(mask):
  # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  dilation = cv2.dilate(mask, kernel)
  erode = cv2.erode(mask, kernel)
  edge = dilation - erode
  return edge


# 4. 设置受限卷积的参数
def set_constrain(weight):
  # 将中心坐标设置为0
  center = int(weight.shape[2] / 2)

  weight[:, :, center, center] = 0
  # 将除了中心坐标的值的和，设置为1
  for i in range(weight.shape[0]):
    for j in range(weight.shape[1]):
      idx_positive = weight[i, j, :, :] >= 0
      idx_negative = weight[i, j, :, :] < 0

      abs_sum = torch.abs(weight[i, j, :, :]).sum()

      if idx_positive.any():
        weight[i, j, idx_positive] = weight[i, j, idx_positive] / abs_sum
        weight[i, j, weight[i, j, :, :] < 0.001] = 0.001

      if idx_negative.any():
        weight[i, j, idx_negative] = 0.001

      weight[i, j, center, center] = -(weight[i, j, :, :].sum())
  return weight


# def set_constrain(weight):
#   # 将中心坐标设置为0
#   center = int(weight.shape[2] / 2)
#
#   weight[:, :, center, center] = 0
#   # 将除了中心坐标的值的和，设置为1
#   for i in range(weight.shape[0]):
#     for j in range(weight.shape[1]):
#       sum = weight[i, j, :, :].sum()
#       weight[i, j, :, :] = weight[i, j, :, :] / sum * 10
#       weight[i, j, center, center] = -10
#   return weight




# 5. 计算f1
@jit(nopython=True)
def cal_f1(mask_pred, mask_gt):
  mask_gt = mask_gt.flatten()
  mask_pred = mask_pred.flatten()

  tp = 0
  fp = 0
  fn = 0

  for i in range(mask_gt.shape[0]):
    if mask_gt[i] == 1 and mask_pred[i] == 1:
      tp += 1
    elif mask_gt[i] == 0 and mask_pred[i] == 1:
      fp += 1
    elif mask_gt[i] == 1 and mask_pred[i] == 0:
      fn += 1
  if tp == 0 and fp == 0 and fn == 0:
    return 0
  f1 = 2 * tp / (2 * tp + fp + fn)
  return f1

@jit(nopython=True)
def cal_all(mask_pred, mask_gt, n_bins=1000):
  mask_gt = mask_gt.flatten()
  mask_pred = mask_pred.flatten()

  tp = 0
  fp = 0
  fn = 0

  # 正样本数量
  postive_len = 0
  # 两类桶
  pos_histogram = [0 for _ in range(n_bins + 1)]
  neg_histogram = [0 for _ in range(n_bins + 1)]
  # 宽度
  bin_width = 1.0 / n_bins

  for i in range(mask_gt.shape[0]):
    if mask_gt[i] == 1 and mask_pred[i] >= 0.5:
      tp += 1
    elif mask_gt[i] == 0 and mask_pred[i] >= 0.5:
      fp += 1
    elif mask_gt[i] == 1 and mask_pred[i] < 0.5:
      fn += 1

    nth_bin = int(mask_pred[i] / bin_width)
    if mask_gt[i] == 1:
      postive_len += 1
      pos_histogram[nth_bin] += 1
    else:
      neg_histogram[nth_bin] += 1

  if tp == 0:
    pre = 0
    rec = 0
    f1 = 0
  else:
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

  accumulated_neg = 0
  satisfied_pair = 0
  for i in range(n_bins + 1):
    satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
    accumulated_neg += neg_histogram[i]

  total_case = postive_len * (mask_gt.shape[0] - postive_len)  # 正负样本对
  auc = satisfied_pair / float(total_case)

  return pre, rec, f1, auc


# 6. 生成据类矩阵
def gen_distance(shape):
  k = shape
  arr = torch.zeros([k * k, k * k], requires_grad=False)
  for i in range(k):
    for j in range(k):
      for x in range(k):
        for y in range(k):
          arr[i * k + j][x * k + y] = math.sqrt((i - x) * (i - x) + (j - y) * (j - y)) + 1

  return arr


# 7. 生成受限卷积的卷积核值
def gen_cons_conv_weight(shape):
  center = int(shape / 2)
  accumulation = 0
  for i in range(shape):
    for j in range(shape):
      if i != center or j != center:
        dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
        accumulation += 1 / dis

  base = 1 / accumulation
  # base = 10 / accumulation
  arr = torch.zeros((shape, shape), requires_grad=False)
  for i in range(shape):
    for j in range(shape):
      if i != center or j != center:
        dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
        arr[i][j] = base / dis
  arr[center][center] = -1

  return arr.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)
  # return arr.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
  # return arr.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)

