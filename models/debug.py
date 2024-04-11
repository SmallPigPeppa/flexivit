import torch
import torch.nn as nn

# 定义一个Conv2d层
conv_layer = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(3, 3), stride=(1, 1))

# 打印权重的形状
print("Weight shape:", conv_layer.weight.shape)

# 如果你想看看bias的形状
print("Bias shape:", conv_layer.bias.shape)
