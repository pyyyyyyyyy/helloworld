import torch
import torchvision
import torchvision.transforms as transforms  # 常用的图片变换
import matplotlib.pyplot as plt
import time
from presentation import *
import sys
import numpy as np

from torch import nn
from torch.nn import init

# # 输出数据集大小
# # print(type(mnist_train))
# # print(len(mnist_train),len(mnist_test))
# #
# # 通过下标访问任意一个样本
# # feature, label =mnist_train[0]
# # print(feature.shape, label)
# #
# # 绘制图样
# # X, y=[], []
# # for i in range(10):
# #    X.append(mnist_train[i][0])
# #    y.append(mnist_test[i][1])
# # show_fashion_mnist(X, get_fashion_minist_labels(y))
# #
# 批量读取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
#
# # # 读取一批数据所需的时间
# # start = time.time()
# # for X, y in train_iter:
# #     continue
# # print('%.2f sec' % (time.time() - start))
#
# 设置模型参数
num_inputs = 784
num_outputs = 10
#
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
# b = torch.zeros(num_outputs, dtype=torch.float)
# # 调整参数梯度
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(requires_grad=True)
#
#
# def net(X):
#     # 通过view将每张原始图修改成长度为num_inputs的向量   1*28*28变成784*1
#     return Softmax(torch.mm(X.view((-1, num_inputs)), w) + b)
#
#
# # y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# # y = torch.LongTensor([0, 2])
# # print(y_hat.gather(1, y.view(-1, 1)))
# #
# #
# # def accuracy(y_hat, y):
# #     # 将y_hat中每行最大的值与y比较，返回一个值为0或1的浮点型tensor
# #     return (y_hat.argmax(dim=1) == y).float().mean().item()
# #
# #
# # print(accuracy(y_hat, y))
# # print(evaluate_acuuracy(test_iter, net))
#
#
# # 训练模型
# num_epochs, lr = 5, 0.1
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)


# # 简洁实现：初始化定义模型   sotfmax输出层是一个全连接层，使用一个线性模块
# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#     def forward(self,x):
#         y = self.linear(x.view(x.shape[0], -1))
#         return y


# 定义网络结构net
from collections import OrderedDict

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))])
)

# 初始化参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
