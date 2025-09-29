from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

##  test tensorboard
writer = SummaryWriter()
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)
writer.close()

##  test item()
# a = torch.tensor([3.14])
# print(a.item())

##  test [:, 0]
# B = 2   # batch size
# C = 3   # feature dimension
# L = 4   # sequence length
# dT = torch.randn(B, C, L)
# lvl_1L = dT[:, 0, :]
# lvl_1L_2 = dT[:, 0]
# print(lvl_1L.shape) # torch.Size([2, 4])
# print(lvl_1L_2.shape) # torch.Size([2, 4])
# print(lvl_1L.equal(lvl_1L_2)) # True

## test torch.autograd.Function
# import torch

# class MyAddBias(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, bias):
#         tmp = input + bias  # 局部变量 tmp，参与了计算
#         return tmp  # 返回结果

#     @staticmethod
#     def backward(ctx, grad_output):
#         # 这里没有保存 tmp，无法在反向传播中使用 tmp
#         grad_input = grad_output.clone()
#         grad_bias = grad_output.clone()
#         return grad_input, grad_bias

# # 示例代码
# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # 输入张量
# bias = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)  # 偏置参数

# # 使用自定义的 MyAddBias 操作
# y = MyAddBias.apply(x, bias)

# # 输出前向结果
# print("Output y:", y)

# # 反向传播
# y.sum().backward()

# # 输出梯度
# print("Gradient with respect to input (x.grad):", x.grad)
# print("Gradient with respect to bias (bias.grad):", bias.grad)

## test drop
# import torch
# import torch.nn as nn

# x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
# drop = nn.Dropout(p=0.5)

# drop.train()  # 启用 Dropout
# print(drop(x))  # 输出中大约有一半元素变成了 0（每次运行随机）

# drop.eval()  # 禁用 Dropout，回到原样
# print(drop(x))  # 输出就是 x 本身

## expand
# x = torch.rand((1,3))
# print(x)
# x = x.expand(4, 3)
# print(x)






