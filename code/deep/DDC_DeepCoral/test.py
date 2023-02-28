import torch

from mmd import MMD_loss


source = torch.rand(64,14)  # 可以理解为源域有64个14维数据
target = torch.rand(32,14)  # 可以理解为源域有32个14维数据
print(target)

MMD = MMD_loss()
a = MMD(source=source, target=target)
print(a)