import matplotlib
import torch
import numpy as np
import matplotlib.pylab as plt
import sys
from presentation import *

from torch import nn
from torch.nn import init

# 损失函数
# # 绘图函数
# def xyplot(x_vals, y_vals, name):
#     set_figsize(figsize=(5, 2.5))
#     plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
#     plt.xlabel('x')
#     plt.ylabel(name + '(x)')
#     plt.show()
#
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = x.relu()
# xyplot(x, y, 'relu')
#
# y.sum().backward()
# xyplot(x, x.grad, 'grad of relu')
#
# y = x.sigmoid()
# xyplot(x, y, 'sigmoid')
#
# x.grad.zero_()
# y.sum().backward()
# xyplot(x, x.grad, 'grad of sigmoid')
#
# y = x.tanh()
# xyplot(x, y, 'tanh')
#
# x.grad.zero_()
# y.sum().backward()
# xyplot(x, x.grad, 'grad of tanh')

# 模型参数
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hidden1s, num_hidden2s = 784, 10, 256, 128
# W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens,)), dtype=torch.float)
# b1 = torch.zeros(num_hiddens,dtype=torch.float)
# W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs,)), dtype=torch.float)
# b2 = torch.zeros(num_outputs, dtype=torch.float)
#
# params = [W1, b1, W2, b2]
# for param in params:
#     # 调整参数梯度
#     param.requires_grad_(requires_grad=True)
#
# def relu(X):
#     return torch.max(input=X, other=torch.tensor(0.0))
#
# def net(X):
#     # 修改原始图片长度
#     X = X.view((-1,num_inputs))
#     H = relu(torch.matmul(X, W1) + b1)
#     return torch.matmul(H, W2) + b2
#
# loss = torch.nn.CrossEntropyLoss()
#
# num_epochs, lr = 5, 100.0
# train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# 简洁实现
net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hidden1s),
    nn.ReLU(),
    nn.Linear(num_hidden1s, num_hidden2s),
    nn.ReLU(),
nn.Linear(num_hidden2s, num_outputs)
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr= 0.1)

num_epochs = 20
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

