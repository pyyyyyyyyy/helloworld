import torch
import numpy as np
import matplotlib
import sys
from presentation import *
import torch.utils.data

# 生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test), 1)
ploy_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)  # torch.cat:拼接  torch.pow:幂运算
labels = (true_w[0] * ploy_features[:, 0] + true_w[1] * ploy_features[:, 1] + true_w[2] * ploy_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 加上噪声

# print(features[: 2], ploy_features[: 2], labels[: 2])

num_epochs, loss = 100, torch.nn.MSELoss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    plt.show()
    print('weight:', net.weight.data, '\nbias:', net.bias.data)


# 正常拟合
# fit_and_plot(ploy_features[:n_train, :], ploy_features[n_train:, :], labels[:n_train], labels[n_train:])

# 欠拟合
# fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])

# 过拟合
fit_and_plot(ploy_features[:10, :], ploy_features[n_train:, :], labels[:10], labels[n_train:])
