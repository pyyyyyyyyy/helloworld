from matplotlib import pyplot as plt
import torch
import random
import numpy as np
from presentation import *
import torch.utils.data as Data  # 提供数据处理工具；读取数据，代替data_iter()
import torch.nn as nn  # 包含大量神经网络层定义
from torch.nn import init   # 初始化模型参数 权重、偏差初始化
import torch.optim as optim     # 包含常用优化算法

# 生成数据集
num_inputs = 2  # n_x
num_examples = 1000  # m
true_w = [2, -3.4]  # w
true_b = 4.2  # b
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)  # 生成服从随机正态分布的数据集（均值，标准差，输出值shape）
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 线性回归  a[:,i]表示取矩阵a的第i列数据
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 加一个随机噪声项

# # print(features[0], labels[0])
#
# # set_figsize()
# # plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# # plt.show()
#
# # 初始化模型参数
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)))
# b = torch.zeros(1)
#
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)
#
# lr = 0.03  # 学习率
# num_epochs = 3  # 迭代周期
# net = linreg  # 模型
# loss = squared_loss  # 损失函数
#
# batch_size = 10
#
# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y).sum()     # 求损失函数（向量化）
#         l.backward()    # 求梯度
#         sgd([w, b], lr, batch_size)     # 优化算法sgd
#
#         w.grad.data.zero_()     # 梯度清零
#         b.grad.data.zero_()
#     train_l = loss(net(features, w, b), labels)
#     print('epoch %d, loss %f' % (epoch+1, train_l.mean().item()))
#
# print(true_w,'\n', w)
# print(true_b,'\n', b)

batch_size = 10
dataset = Data.TensorDataset(features, labels)  # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 随机小批量读取


class LinearNet(nn.Module):  # 继承nn.Module，撰写自己的网络层(包含层数和前向传播方法)
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)
print(net)    # 打印网络结构
print(net[0])

# 初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)

# 定义损失函数(均方误差)
loss = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
# print(optimizer)

# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1    # 学习率为原来的0.1倍

# 训练模型
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()   # 清除梯度
        l.backward()
        optimizer.step()
    print('epoch %d, loss %f' % (epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)