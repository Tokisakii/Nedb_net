import cv2
import torch
import datetime
import re
import os
import utils
from utils import log as log
from config import Config
from torch.nn import functional as F
from criterion import CrossEntropy, OhemCrossEntropy, BceEntropy, DiceLoss
import collections
import numpy as np
import random

# 保存当前已训练的epoch数
epoch = 0
# 保存模型的路径
checkpoint_path = None


# 0. 随机数种子
def worker_init_fn(worker_id):
  worker_seed = torch.initial_seed() % 2 ** 32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


# 1. 加载模型参数
def load_pre_weights(model, filepath):
  if os.path.exists(filepath):
    pretrained_dict = torch.load(filepath)
    model_dict = model.state_dict()

    # 如果加载的是resnet，则需要将权重的key进行处理才能加载到自己的模型
    if filepath.startswith("./resnet"):
      filtered_dict = collections.OrderedDict()
      del pretrained_dict['fc.weight']
      del pretrained_dict['fc.bias']

      for k, v in pretrained_dict.items():
        if k.startswith("layer3") or k.startswith("layer4"):
          layersl = k[0:6] + "l" + k[6:]
          layersh = k[0:6] + "h" + k[6:]
          filtered_dict["module." + layersl] = v
          filtered_dict["module." + layersh] = v
        else:
          filtered_dict["module." + k] = v
        pretrained_dict = filtered_dict

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
  else:
    print("Weight file not found ...")


# 3. 设置模型保存路径，同时根据模型名字设定训练的epoch数
def set_model_dir(model_path):
  # 1. 两个全局变量
  global checkpoint_path
  global epoch

  # 2. 获得当前时间
  now = datetime.datetime.now()

  # 3. 根据模型路径名称获得模型训练的epoch数以及时间
  if model_path:
    regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/ddrnet\_\w+(\d{4})\.pth"
    m = re.match(regex, model_path)
    if m:
      now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                              int(m.group(4)), int(m.group(5)))
      epoch = int(m.group(6))

  # 4. 结合时间，设置模型保存目录
  # 默认是 "./logs/ddrnet23"； "./logs/ddrnet23/ddrnet_casia2_*epoch*.pth"； "./logs/ddrnet23/ddrnet_casia2_{:04d}.pth"
  log_dir = os.path.join(Config.model_dir, "{}{:%Y%m%dT%H%M}".format(Config.dataset_name.lower(), now))
  checkpoint_path = os.path.join(log_dir, "ddrnet_{}_*epoch*.pth".format(Config.dataset_name.lower()))
  checkpoint_path = checkpoint_path.replace("*epoch*", "{:04d}")
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  return checkpoint_path


# 4. 根据配置得到损失函数。后边没有用到
def get_criterion(class_weights):
  if Config.use_ohem:
    criterion = OhemCrossEntropy(thres=Config.ohem_thres,
                                 min_kept=Config.ohem_kept,
                                 weight=class_weights)
  elif Config.use_bec:
    criterion = BceEntropy()
  else:
    criterion = CrossEntropy(weight=class_weights)
  return criterion


# 5. 根据配置得到优化器
def get_optimizer(model, learning_rate):
  params_dict = dict(model.named_parameters())
  params = [{'params': list(params_dict.values()), 'lr': learning_rate}]
  optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=Config.momentum, weight_decay=Config.weight_decay)
  return optimizer


# 6. 训练
def train(model, train_dataset, learning_rate, epochs, batchsize, steps):
  # 1. 生成DataLoader
  # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
  # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

  # 2. 准备开始训练
  # 2.1 打印当前epoch、学习率、检查点路径
  global epoch, checkpoint_path
  log("\nStarting at epoch {}. LR={}\n".format(epoch, learning_rate))
  log("Checkpoint Path: {}".format(checkpoint_path))

  # 2.2 设置损失函数、optimizer、开启训练模式
  # criterion = get_criterion(train_dataset.class_weights)
  optimizer = get_optimizer(model, learning_rate)
  model.train()

  bce_criterion = torch.nn.BCELoss()
  dice_loss = DiceLoss()
  # 3. 开始训练，循环指定个epoch
  for epoch in range(epoch + 1, epochs + 1):
    # 打印当前epoch数和总epoch数；将当前epoch的loss和step重置为0
    log("Epoch {}/{}.".format(epoch, epochs))
    epoch_loss = 0
    step = 0

    # 循环拿出batch size大小的数据
    for inputs in train_dataloader:
      # 1. 获取当前iter的数据，并丢到GPU
      images, labels, edges, names = inputs
      # print(names)
      images = images.float().cuda()
      labels = labels.cuda()
      edges = edges.cuda()

      # 2. 开始预测，计算损失，反向传播
      masks_pred, edges_pred = model(images)
      masks_pred = torch.sigmoid(masks_pred)
      edges_pred = torch.sigmoid(edges_pred)

      masks_loss_bce = bce_criterion(masks_pred, labels.unsqueeze(1).float())
      masks_loss_dice = dice_loss(masks_pred, labels.unsqueeze(1).float())

      edges_loss_bce = bce_criterion(edges_pred, edges.unsqueeze(1).float())
      edges_loss_dice = dice_loss(edges_pred, edges.unsqueeze(1).float())

      # 边缘的损失占0.8，篡改区域的损失占0.2。与论文不同的是，因为加入了非篡改的数据集，所以加入了bec_loss，每一类损失由bce_loss和dice_loss共同组成
      loss = (masks_loss_bce * 0.2 + masks_loss_dice * 0.8) * 0.2 + (edges_loss_bce * 0.2 + edges_loss_dice * 0.8) * 0.8

      model.zero_grad()
      loss.backward()
      optimizer.step()

      # 2.2 受限卷积修改3，设置受限卷积的权重
      with torch.no_grad():
        constrained_conv_weight = model.module.cons_conv.weight.data.clone().detach()
        constrained_conv_weight = utils.set_constrain(constrained_conv_weight)
        model.module.cons_conv.weight.copy_(constrained_conv_weight)

      # 3. 把这个iter的损失加入到本epcoh的总损失里，并打印结果
      epoch_loss += loss.item()
      print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, step, steps, loss.item()))
      # 4. iter数加1，如果达到steps说明，这个epoch结束了。打印结果，保存模型
      step = step + 1
      if step % steps == 0:
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / steps))
        with open("train_result.txt", "a") as f:
          f.writelines(str(epoch) + "\n")
          f.writelines(str(epoch_loss / steps) + "\n")

        # 保存模型，这里设置了epoch > 40时才保存
        # if epoch > 40:
        #   torch.save(model.state_dict(), checkpoint_path.format(epoch))
        #   break

        # 保存模型
        torch.save(model.state_dict(), checkpoint_path.format(epoch))
        break

  # 4. 保存本次训练了多少个epoch
  epoch = epochs


# 7. 预测
def inference(model, image):
  model.eval()
  with torch.no_grad():
    # 1. 将模型缩放到设置大小，并转成tensor
    image_resize = cv2.resize(image, (Config.base_size, Config.base_size))
    image_resize = utils.image_to_tensor(image_resize, Config.mean, Config.std)

    # 2. 得到模型篡改区域和边缘的预测结果；将结果恢复到图像原始大小；sigmoid之后默认0.5阈值二值化
    mask_pred, edge_pred = model(image_resize)
    mask_pred = F.interpolate(input=mask_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=Config.align_corners)
    mask_pred = torch.sigmoid(mask_pred)
    mask_pred = (mask_pred.squeeze(0).squeeze(0) >= 0.5).long()

    edge_pred = F.interpolate(input=edge_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=Config.align_corners)
    edge_pred = torch.sigmoid(edge_pred)
    edge_pred = (edge_pred.squeeze(0).squeeze(0) >= 0.5).long()

    return mask_pred.cpu().numpy(), edge_pred.cpu().numpy()










