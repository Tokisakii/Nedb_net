import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


# 1. 边缘融合模块，借鉴了SKNet。
class EMB(nn.Module):
  # M表示要融合的边缘预测数量，默认是2
  def __init__(self, features, M=2):
    super(EMB, self).__init__()
    d = int(features / 2)
    self.M = M
    self.features = features

    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(d),
                            nn.ReLU(inplace=True))
    # 有几个待融合的特征就构建几个卷积操作
    self.fcs = nn.ModuleList([])
    for i in range(M):
      self.fcs.append(
        nn.Conv2d(d, features, kernel_size=1, stride=1)
      )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, f1, f2, f3=None, f4=None):
    batch_size = f1.shape[0]

    # 1. 将待融合的边缘预测特征在channel维度进行拼接。这里只写了边缘预测数量是2和4的情况，代码写的不好
    if self.M == 2:
      feats = torch.cat((f1, f2), dim=1)
    else:
      feats = torch.cat((f1, f2, f3, f4), dim=1)

    # 2. 将维度变为[batch_size, 待融合的边缘预测数量, channel数, 高, 宽]
    feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

    # 3. 求和，平均池化，卷积预测
    feats_U = torch.sum(feats, dim=1)
    feats_S = self.gap(feats_U)
    feats_Z = self.fc(feats_S)

    # 4. 使用不同的卷积对特征进行学习，分别达到每一个待融合特征自己的融合权重
    attention_vectors = [fc(feats_Z) for fc in self.fcs]
    attention_vectors = torch.cat(attention_vectors, dim=1)
    attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
    attention_vectors = self.softmax(attention_vectors)

    # 5. 待融合特征进行融合
    feats_V = torch.sum(feats * attention_vectors, dim=1)

    return feats_V


# 2. 加入距离因素的Non-local模块
class NonLocalBlock(nn.Module):
  def __init__(self, in_channels, inter_channels=1, shape=32):
    super(NonLocalBlock, self).__init__()

    self.in_channels = in_channels
    self.inter_channels = in_channels // 4
    inter_channels = in_channels // 4
    conv_nd = nn.Conv2d

    self.w_v = nn.Sequential(
      conv_nd(in_channels=in_channels, out_channels=in_channels,
              kernel_size=1, stride=1, padding=0, bias=False),
      # self.bn)
    )

    self.w_q = nn.Sequential(
      conv_nd(in_channels=in_channels, out_channels=inter_channels,
              kernel_size=1, stride=1, padding=0, bias=False),
      # bn_inter)
    )

    self.w_k = nn.Sequential(
      conv_nd(in_channels=in_channels, out_channels=inter_channels,
              kernel_size=1, stride=1, padding=0, bias=False),
      # bn_inter)
    )

    self.w_upper_channel = nn.Sequential(
      nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(in_channels)
    )

    self.distance = utils.gen_distance(shape).cuda()
    self.relu = nn.ReLU()

  def forward(self, x):
    batch_size = x.size(0)

    # 使用高分辨率的feature，1x1卷积降维，然后reshape，然后再把channel换到最后，即 [batch, h*w, inter_channels]，得到v
    v = self.w_v(x).view(batch_size, self.in_channels, -1)
    v = v.permute(0, 2, 1)

    # 使用低分辨率的feature，形成q [batch, h*w, inter_channels] 和 k [batch, inter_channels, h*w]
    q = self.w_q(x).view(batch_size, self.inter_channels, -1)
    q = q.permute(0, 2, 1)
    k = self.w_k(x).view(batch_size, self.inter_channels, -1)

    # 将q和k相乘得到 [batch, h*w, h*w]，除以距离矩阵后，进行softmax表示一个全局的注意力
    relation = torch.matmul(q, k)
    relation = relation / self.distance
    relation = F.softmax(relation, dim=-1)
    # relation = torch.sigmoid(relation)

    # 将注意力缩放到高分辨率大小，和v相乘，交换维度再将h*w拆开，并且升维到原始，得到 [batch, in_channels, h, w]
    y = torch.matmul(relation, v)
    y = y.permute(0, 2, 1).contiguous()
    y = y.view(batch_size, self.in_channels, *x.size()[2:])

    return self.relu(self.w_upper_channel(y) + x)


# 3. 普通的卷积块
class ConvBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()

  def forward(self, input):
    x = self.conv1(input)
    return self.relu(self.bn(x))


# 4. BiSeNet的ARM块
class AttentionRefinementModule(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

  def forward(self, input):
    x = self.avgpool(input)
    assert self.in_channels == x.size(1)
    x = self.conv(x)
    x = self.bn(x)
    x = self.sigmoid(x)
    x = torch.mul(input, x)

    x = self.bn(x)
    x = self.relu(x)
    return x


# 5. BiSeNet的FFM块
class FeatureFusionModule(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=self.out_channels, stride=1)
    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    self.sigmoid = nn.Sigmoid()
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

  def forward(self, input_1, input_2):
    x = torch.cat((input_1, input_2), dim=1)
    assert self.in_channels == x.size(1)
    feature = self.convblock(x)

    x = self.avgpool(feature)
    x = self.relu(self.bn(self.conv1(x)))
    x = self.sigmoid(self.bn(self.conv2(x)))
    x = torch.mul(feature, x)

    x = torch.add(x, feature)

    x = self.bn(x)
    x = self.relu(x)
    return x


# 6. 边缘提取块。先对特征进行降维，然后再构建残差学习
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


# 7. 构建ResNet34的BasicBlock
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


# 0. 模型整体的结构，基于ResNet34修改
class ResNet(nn.Module):
  # 1. 模型参数
  def __init__(self, block, layers, num_classes=1000):
    # 0. 构建ResNet
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
    self.cons_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, bias=False, padding=2)

    # 2. 构建双分支网络的高分辨率分支。stride都设置成1
    self.inplanes = 128
    self.layer3h = self._make_layer(block, 256, layers[2], stride=1)
    self.layer4h = self._make_layer(block, 512, layers[3], stride=1)

    # 3. 6个边缘提取块
    self.erb1 = ERB(64, 32, 1)
    self.erb2 = ERB(128, 32, 2)
    self.erb3_h = ERB(256, 32, 4)
    self.erb4_h = ERB(512, 32, 8)
    self.erb3_l = ERB(256, 32, 4)
    self.erb4_l = ERB(512, 32, 8)

    # 4. 3个边缘融合块
    self.emb3lh = EMB(32)
    self.emb4lh = EMB(32)
    self.emb = EMB(32, 4)

    # 5. 用来得到最后的篡改边缘预测结果
    self.erb_merge = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(1))

    # 6. 两个arm模块和一个ffm模块。其中ffm模块用来融合高分辨率分支和上下文分分支的feature
    self.arm1 = AttentionRefinementModule(256, 256)
    self.arm2 = AttentionRefinementModule(512, 512)
    self.ffm = FeatureFusionModule(256 + 512, 256)

    # 7. 两个non-local模块
    self.non_local1 = NonLocalBlock(256, shape=32)
    self.non_local2 = NonLocalBlock(512, shape=16)

    # 8. 上下文分支和高分辨率分支融合时，和论文不太相同。先将上下文分支3l和4l在channel维度拼接后，使用一个卷积进行处理。再和高分辨率分支的4h使用ffm进行融合
    self.block_merge = nn.Sequential(nn.Conv2d(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     self.relu)

    # 9. 上下文分支和高分辨率分支融合，后续使用3个卷积预测最终的篡改mask
    self.down_channel1 = nn.Sequential(
      nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(64),
      self.relu
    )
    self.down_channel2 = nn.Sequential(
      nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(16),
      self.relu
    )
    self.down_channel3 = nn.Sequential(
      nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(1)
    )

    # 10. 设置模型初值
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

    # 11. 将受限卷积单独初始化
    with torch.no_grad():
      self.cons_conv.weight.copy_(utils.gen_cons_conv_weight(5))

  # 2. 构建基本ResNet用到
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

  # 3. 模型过程
  def forward(self, x):
    # 1. 原始输入经过受限卷积提取早上图像
    x = self.cons_conv(x)

    # 2. 经过ResNet的前两层，并使用erb提取对应的边缘edge1和edge2
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    edge1 = self.erb1(x)

    x = self.layer2(x)
    edge2 = self.erb2(x)
    # 为了和edge1的大小相同，需要对edge2进行上采样，后续的预测边缘也需要上采样操作
    edge2 = F.interpolate(edge2, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)


    # 3. 经过高分辨率3h层和上下文3l层。从xh和xl中提取两个边缘edge3_h和edge3_l。对第上下文3l层的特征xl分别用arm和non-local处理，留着后续融合使用。
    xh = self.layer3h(x)
    xl = self.layer3l(x)
    xl_arm1 = self.arm1(xl)
    non_local1 = self.non_local1(xl_arm1)
    non_local1 = F.interpolate(non_local1, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)

    edge3_h = self.erb3_h(xh)
    edge3_h = F.interpolate(edge3_h, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    edge3_l = self.erb3_l(xl)
    edge3_l = F.interpolate(edge3_l, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)


    # 4. 经过高分辨率4h层和上下文4l层。从xh和xl中提取两个边缘edge4_h和edge4_l。对第上下文4l层的特征xl分别用arm和non-local处理，留着后续融合使用。
    xh = self.layer4h(xh)
    xl = self.layer4l(xl)
    xl_arm2 = self.arm2(xl)
    # xl_arm2 = xl
    non_local2 = self.non_local2(xl_arm2)
    non_local2 = F.interpolate(non_local2, scale_factor=4, mode='bilinear', align_corners=Config.align_corners)

    # edge4 = self.erb4(xh)
    edge4_h = self.erb4_h(xh)
    edge4_h = F.interpolate(edge4_h, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    edge4_l = self.erb4_l(xl)
    edge4_l = F.interpolate(edge4_l, scale_factor=8, mode='bilinear', align_corners=Config.align_corners)


    # 5. 融合边缘预测结果。先将edge3_l和edge3_h融合为edge3。再将先将edge4_l和edge4_h融合为edge4。最后融合edge1-4。
    edge3 = self.emb3lh(edge3_l, edge3_h)
    edge4 = self.emb4lh(edge4_l, edge4_h)
    edge = self.emb(edge1, edge2, edge3, edge4)
    # 使用一个卷积得到最后的篡改边缘预测结果
    edge = self.erb_merge(edge)

    # 6. 融合时上下文分支和高分辨率分支的特征，和论文不太相同。
    # 先将上下文分支3l和4l在channel维度拼接后，使用一个卷积进行处理。再和高分辨率分支的4h使用ffm进行融合
    x_merge = self.ffm(xh, self.block_merge(torch.cat((non_local2, non_local1), dim=1)))


    # 7. 将融合后的特征上采样，并且使用卷积减少channel数量。最后恢复成原始输入 batch*1*512*512大小。
    # 1/4
    x = F.interpolate(x_merge, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel1(x)

    # 1/2
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel2(x)

    # 1/1
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=Config.align_corners)
    x = self.down_channel3(x)

    return x, edge


def get_seg_model(cfg):
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  return model

