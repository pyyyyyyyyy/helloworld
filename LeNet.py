#!/usr/bin/env python
# coding: utf-8

# In[6]:


import time
import torch
from torch import nn,optim
from presentation import *
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[65]:


class LENet(nn.Module):
    def __init__(self):
        super(LENet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
    
    def forward(self, img):
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0], -1))

net = LENet()
print(net)
                                  


# In[63]:


# 下载数据集 torchvision.datasets：加载数据的函数和常用的的数据及接口
mnist_train = torchvision.datasets.FashionMNIST(root='D:\Repos\PycharmProjects\Datasets\FashionMNIST',
                                                train=True, download=True,
                                                transform=transforms.ToTensor())  # 将所有数据转换成Tensor (C,H,W)
mnist_test = torchvision.datasets.FashionMNIST(root='D:\Repos\PycharmProjects\Datasets\FashionMNIST',
                                               train=False, download=True, transform=transforms.ToTensor())
# print(len(mnist_train),len(mnist_test))

# feature, lable = mnist_train[20]
# print(feature,lable)


# In[64]:


batch_size = 256
num_epochs = 10

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

loss = nn.CrossEntropyLoss()

lr = 0.001
optimizer = torch.optim.Adam(net.parameters(),lr=lr)


# In[66]:


def evaluate_accuracy(test_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n
        


# In[67]:


def train(net, train_iter, test_iter, optimizer, loss, device, num_epochs):
    # 设置GPU训练
    net = net.to(device)
    print("training on ",device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            # 设置GPU训练
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'% (epoch+1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time()-start))
            


# In[ ]:


train(net, train_iter, test_iter, optimizer, loss, device, num_epochs)


# In[ ]:




