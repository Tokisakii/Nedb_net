import torch.nn as nn
from torch.nn import functional as F
from config import Config


# 0. DiceLoss实现
class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()

  def forward(self, input, target):
    N = target.size(0)
    smooth = 1e-6

    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N

    return loss


# 0. BCE的实现
class BceEntropy(nn.Module):
  def __init__(self):
    super(BceEntropy, self).__init__()
    self.criterion = nn.BCELoss()

  def _forward(self, score, target):
    ph, pw = score.size(2), score.size(3)
    h, w = target.size(2), target.size(3)
    if ph != h or pw != w:
      score = F.interpolate(input=score, size=(
        h, w), mode='bilinear', align_corners=Config.align_corners)

    score = F.sigmoid(score)
    loss = self.criterion(score, target.float())
    return loss

  def forward(self, score, target):
    if Config.num_output == 1:
      score = [score]

    # 增加一个channel维度
    target = target.unsqueeze(1)

    # 网络可能不止最后一个输出，这里是每个输出的loss占比
    weights = Config.balance_weights
    assert len(weights) == len(score)

    return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])



# 1. CE的实现
class CrossEntropy(nn.Module):
  def __init__(self, ignore_label=-1, weight=None):
    super(CrossEntropy, self).__init__()
    self.ignore_label = ignore_label
    # 不同类别的loss占比
    self.criterion = nn.CrossEntropyLoss(
      weight=weight,
      ignore_index=ignore_label
    )

  def _forward(self, score, target):
    ph, pw = score.size(2), score.size(3)
    h, w = target.size(1), target.size(2)
    if ph != h or pw != w:
      score = F.interpolate(input=score, size=(
        h, w), mode='bilinear', align_corners=Config.align_corners)

    loss = self.criterion(score, target)

    return loss

  def forward(self, score, target):

    if Config.num_output == 1:
      score = [score]

    # 网络可能不止最后一个输出，这里是每个输出的loss占比
    weights = Config.balance_weights
    assert len(weights) == len(score)

    return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


# 2. OHEM，thresh为损失函数大于多少的时候，会被用来做反向传播。n_min为在一个 batch 中，最少需要考虑多少个样本。
class OhemCrossEntropy(nn.Module):
  def __init__(self, ignore_label=-1, thres=0.7,
               min_kept=100000, weight=None):
    super(OhemCrossEntropy, self).__init__()
    self.thresh = thres
    self.min_kept = max(1, min_kept)
    self.ignore_label = ignore_label
    self.criterion = nn.CrossEntropyLoss(
      weight=weight,
      ignore_index=ignore_label,
      reduction='none'
    )

  def _ce_forward(self, score, target):
    ph, pw = score.size(2), score.size(3)
    h, w = target.size(1), target.size(2)
    if ph != h or pw != w:
      score = F.interpolate(input=score, size=(
        h, w), mode='bilinear', align_corners=Config.align_corners)

    loss = self.criterion(score, target)

    return loss

  def _ohem_forward(self, score, target, **kwargs):
    ph, pw = score.size(2), score.size(3)
    h, w = target.size(1), target.size(2)
    if ph != h or pw != w:
      score = F.interpolate(input=score, size=(
        h, w), mode='bilinear', align_corners=Config.align_corners)
    pred = F.softmax(score, dim=1)
    pixel_losses = self.criterion(score, target).contiguous().view(-1)
    mask = target.contiguous().view(-1) != self.ignore_label

    tmp_target = target.clone()
    tmp_target[tmp_target == self.ignore_label] = 0
    pred = pred.gather(1, tmp_target.unsqueeze(1))
    pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
    min_value = pred[min(self.min_kept, pred.numel() - 1)]
    threshold = max(min_value, self.thresh)

    pixel_losses = pixel_losses[mask][ind]
    pixel_losses = pixel_losses[pred < threshold]
    return pixel_losses.mean()

  def forward(self, score, target):

    if Config.num_output == 1:
      score = [score]

    weights = Config.balance_weights
    assert len(weights) == len(score)

    functions = [self._ce_forward] * \
                (len(weights) - 1) + [self._ohem_forward]
    return sum([
      w * func(x, target)
      for (w, x, func) in zip(weights, score, functions)
    ])
