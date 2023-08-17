import numpy as np
import matplotlib.pyplot as plt


# 画图函数
def display(image, mask_gt, mask_pred, edge_pred=None, figsize=(16,16)):
  # 1. 创建axes
  if edge_pred is None:
    _, axes = plt.subplots(3, 1, figsize=figsize)
  else:
    _, axes = plt.subplots(4, 1, figsize=figsize)

  # 2. 画原图像、对应的mask、mask的预测、边缘的预测
  height, width = image.shape[:2]
  axes[0].set_ylim(height, 0)
  axes[0].set_xlim(0, width)
  axes[0].axis('off')
  axes[0].imshow(image)

  if mask_gt is not None:
    height, width = mask_gt.shape[:2]
    axes[1].set_ylim(height, 0)
    axes[1].set_xlim(0, width)
    axes[1].axis('off')
    axes[1].imshow(mask_gt)

  height, width = mask_pred.shape[:2]
  axes[2].set_ylim(height, 0)
  axes[2].set_xlim(0, width)
  axes[2].axis('off')
  axes[2].imshow(mask_pred)

  if edge_pred is not None:
    height, width = mask_pred.shape[:2]
    axes[3].set_ylim(height, 0)
    axes[3].set_xlim(0, width)
    axes[3].axis('off')
    axes[3].imshow(edge_pred)

  plt.show()
