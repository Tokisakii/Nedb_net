import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

# todo 记得删除
class Config:
  align_corners = True

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


# 基础的ResNet34上增加原始的受限卷积，增加EEB、EMB。第四章的最终结果
class EMB(nn.Module):
  def __init__(self, features, M=2):
    super(EMB, self).__init__()
    d = int(features / 2)
    self.M = M
    self.features = features

    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(d),
                            nn.ReLU(inplace=True))
    self.fcs = nn.ModuleList([])
    for i in range(M):
      self.fcs.append(
        nn.Conv2d(d, features, kernel_size=1, stride=1)
      )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, f1, f2, f3=None, f4=None):
    batch_size = f1.shape[0]
    if self.M == 2:
      feats = torch.cat((f1, f2), dim=1)
    else:
      feats = torch.cat((f1, f2, f3, f4), dim=1)

    feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

    feats_U = torch.sum(feats, dim=1)
    feats_S = self.gap(feats_U)
    feats_Z = self.fc(feats_S)

    attention_vectors = [fc(feats_Z) for fc in self.fcs]
    attention_vectors = torch.cat(attention_vectors, dim=1)
    attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
    attention_vectors = self.softmax(attention_vectors)

    feats_V = torch.sum(feats * attention_vectors, dim=1)

    return feats_V


class ERB(nn.Module):
  def __init__(self, in_channels, out_channels, inter_scale=4):
    super(ERB, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, int(in_channels / inter_scale), kernel_size=1, stride=1, padding=0)

    self.conv2 = nn.Conv2d(int(in_channels / inter_scale), int(in_channels / inter_scale), kernel_size=3, stride=1,
                           padding=1)
    self.relu = nn.ReLU()
    self.bn2 = nn.BatchNorm2d(int(in_channels / inter_scale))
    self.conv3 = nn.Conv2d(int(in_channels / inter_scale), int(in_channels / inter_scale), kernel_size=3, stride=1,
                           padding=1)
    self.bn3 = nn.BatchNorm2d(int(in_channels / inter_scale))

    self.conv4 = nn.Conv2d(int(in_channels / inter_scale), out_channels, kernel_size=1, stride=1, padding=0)
    self.bn4 = nn.BatchNorm2d(out_channels)

  def forward(self, x, relu=True):
    x = self.conv1(x)

    res = self.conv2(x)
    res = self.bn2(res)
    res = self.relu(res)
    res = self.conv3(res)
    res = self.bn3(res)
    res = self.relu(res)

    x = self.conv4(x + res)
    if relu:
      return self.relu(self.bn4(x))
    else:
      return self.bn4(x)


class BasicBlock(nn.Module):
  expansion = 1
  __constants__ = ['downsample']

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3l = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4l = self._make_layer(block, 512, layers[3], stride=2)  # different

    # 1. 受限卷积
    # self.cons_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, bias=False, padding=2)

    # 2. EEB和EMB
    self.erb1 = ERB(64, 32, 1)
    self.erb2 = ERB(128, 32, 2)
    self.erb3_l = ERB(256, 32, 4)
    self.erb4_l = ERB(512, 32, 8)

    self.emb = EMB(32, 4)

    # 边缘融合
    self.erb_merge = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(1))

    # 3. 上采样加卷积预测最终篡改区域
    self.down_channel1 = nn.Sequential(
      nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(128),
      self.relu
    )
    self.down_channel2 = nn.Sequential(
      nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      self.relu
    )
    self.down_channel3 = nn.Sequential(
      nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(1)
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    # todo 将受限卷积单独初始化
    # with torch.no_grad():
    #   self.cons_conv.weight.copy_(utils.gen_cons_conv_weight(5))

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    # 1. 原始输入经过受限卷积提取早上图像
    # x = self.cons_conv(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    edge1 = self.erb1(x)
    x = self.layer2(x)
    edge2 = self.erb2(x)
    edge2 = F.interpolate(edge2, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.layer3l(x)
    edge3 = self.erb3_l(x)
    edge3 = F.interpolate(edge3, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)
    x = self.layer4l(x)
    edge4 = self.erb4_l(x)
    edge4 = F.interpolate(edge4, scale_factor=8, mode='bilinear', align_corners=Config.align_corners)
    edge = self.emb(edge1, edge2, edge3, edge4)
    edge = self.erb_merge(edge)

    # 1/8 512
    x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel1(x)

    # 1/2 128
    x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel2(x)

    # 1/1
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel3(x)

    return x, edge


def get_seg_model(cfg):
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  return model

