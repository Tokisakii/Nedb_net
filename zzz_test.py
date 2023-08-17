# 还有测试没改完
import torch
import numpy as np
if __name__ == '__main__':
    print("hello world11")
    a = torch.tensor(np.array([[[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]]])).float()
    print(a.shape)
    BN = torch.nn.BatchNorm2d(1)
    print(BN(a))