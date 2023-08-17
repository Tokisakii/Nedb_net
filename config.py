# 配置类
class Config(object):
  # 保存模型相关的根目录
  model_dir = "./logs"
  # 模型名字
  dataset_name = "casia2"

  # 数据集类别数
  num_classes = 1
  # 输入图像是否增强
  flip = True
  aug = True
  # 基本resize大小
  base_size = 512
  # 数据集均值
  mean = [0.406, 0.456, 0.485]
  # 数据集方差
  std = [0.225, 0.224, 0.229]


  # 插值是否用align_corners
  align_corners = True

  # 模型有几个输出（可能会输出中间结果也求损失），以及哪个是最终输出的结果
  num_output = 2
  output_index = 0
  # 两个结果分别占损失的比重
  # balance_weights = [1, 0.4]
  balance_weights = [1]

  momentum = 0.9
  weight_decay = 0.0005

  # use_bec = False
  # # 使用ohem损失
  # use_ohem = False
  # ohem_thres = 0.9
  # ohem_kept = 262144
