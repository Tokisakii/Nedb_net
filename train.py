import torch
from config import Config
import casia2_dataset
import nedbnet
import functions as functions

import numpy as np
import random
import os


# 设置随机数种子
def set_seed(seed=0):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  np.random.seed(seed)  # Numpy module.
  random.seed(seed)  # Python random module
  # torch.set_deterministic(True)
  # torch.backends.cudnn.enabled = False
  # torch.backends.cudnn.benchmark = False
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  os.environ['PYTHONHASHSEED'] = str(seed)


class args:
  # todo 1.数据集根目录
  root = "/home/kxg/zzz/datasets/acc_data/"
  # todo 2.训练文件，记录了训练图像的名字
  list_path = "train_all.txt"
  # todo 3.预加载的模型权重文件路径。如果用resnet34.pth则重新训练，如果使用ddrnet_casia2_0040.pth，则会从0040这个epoch继续训练
  # model_path = "./logs/casia220221020T2011/ddrnet_casia2_0040.pth"
  model_path = "./resnet34.pth"
  # 训练的模型权重保存目录
  logs = "./logs"
  # 学习率，main函数自己设置也行
  lr = 0.01
  # todo 4.batchs ize大小
  batchsize = 46
  # todo 5.这里的数字需要和训练文件的文件数相同。
  steps = int(4525 / batchsize)
  gpus = [0]


if __name__ == '__main__':
  # 固定随机数种子
  set_seed(2022)
  Config.use_bec = True

  # 0. 得到参数
  lr = float(args.lr)
  batchsize = int(args.batchsize)
  steps = int(args.steps)

  # 1. 创建数据集
  dataset = casia2_dataset.Casia2Dataset(args.root, args.list_path)

  # 2. 创建模型
  model = nedbnet.get_seg_model(Config)
  model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

  # 3. 根据设置载入参数
  functions.load_pre_weights(model, args.model_path)

  # 4. 设置模型保存路径，同时根据模型名字设定训练的epoch数
  functions.set_model_dir(args.model_path)

  # 5. 训练，不同epoch的学习率
  functions.train(model, dataset, learning_rate=lr, epochs=80, batchsize=batchsize, steps=steps)
  functions.train(model, dataset, learning_rate=0.0075, epochs=100, batchsize=batchsize, steps=steps)
  functions.train(model, dataset, learning_rate=0.005, epochs=130, batchsize=batchsize, steps=steps)
  functions.train(model, dataset, learning_rate=0.0025, epochs=160, batchsize=batchsize, steps=steps)
  functions.train(model, dataset, learning_rate=0.0010, epochs=200, batchsize=batchsize, steps=steps)