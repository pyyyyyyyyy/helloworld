from IPython import display
from matplotlib import pyplot as plt
import random
import torch
import torch.utils.data
import sys
import torchvision
import torchvision.transforms as transforms  # 常用的图片变换
from torch import nn
from torch.nn import init

def use_svg_display():
    # 绘制矢量图
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    # 设置图片尺寸
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    # 读取小批量数据样本，每次返回batch_size个随机数据样本和标签
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机读取样本
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一批不足一个batch_size大小
        yield features.index_select(0, j), labels.index_select(0, j)  # select_index：对tensor实现索引。0：按行，j:索引第j行


# 定义模型(线性回归)
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 定义损失函数(平方差损失函数)
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法(小批量随机梯度下降)
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# Fashion_MNIST中标签值与文本的转化
def get_fashion_minist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 在一行中画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)  # 标题
        f.axes.get_xaxis().set_visible(False)  # x轴不可视
        f.axes.get_yaxis().set_visible(False)  # y轴不可视
    plt.show()


# 下载数据集 torchvision.datasets：加载数据的函数和常用的的数据及接口
mnist_train = torchvision.datasets.FashionMNIST(root='D:\Repos\PycharmProjects\Datasets\FashionMNIST',
                                                train=True, download=True,
                                                transform=transforms.ToTensor())  # 将所有数据转换成Tensor (C,H,W)
mnist_test = torchvision.datasets.FashionMNIST(root='D:\Repos\PycharmProjects\Datasets\FashionMNIST',
                                               train=False, download=True, transform=transforms.ToTensor())


def load_data_fashion_mnist(batch_size):
    # 使用多进程加速读取数据
    if sys.platform.startswith('win'):
        num_workers = 0  # 不使用额外的进程加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


# 对输出数据进行softmax运算
def Softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


# 评价在模型net上的准确率
def evaluate_acuuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # 判断net是否为torch.nn.Module中的一个类别实例
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式，关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 返回训练模式
        else:
            if('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                # 将is_training修改为false
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            # 反向传播
            l.backward()
            # 优化算法、梯度下降
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()


            # 计算总损失、评估准确率
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        # 模型测试
        test_acc = evaluate_acuuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_acc_sum / n, train_acc_sum / n, test_acc))


# 对x实现形状转换,扁平化
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self,x):
        # 将x shape：（batch,*，*，*）   变成（batch，*）
        return x.view(x.shape[0], -1)


# 作图函数
def semilogy(x_vals, y_vals, x_label, y_label,x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        # 图例
        plt.legend(legend)