import torch
from torch import nn, optim
import torch.utils.data
import torchvision
from torchvision import transforms
import time

# train_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(0.5, 0.5),
#     # 左右翻转
#     torchvision.transforms.RandomHorizontalFlip(),
#     # 随机裁剪
#     torchvision.transforms.RandomCrop(28, padding=4)
#
# ])
#
# val_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(0.5, 0.5)
# ])

train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

mnist_train = torchvision.datasets.MNIST(root="D:\Repos\PycharmProjects\Datasets\MNIST", download=True, transform=train_transform, train=True)
mnist_val = torchvision.datasets.MNIST(root="D:\Repos\PycharmProjects\Datasets\MNIST", download=True, transform=val_transform, train=False)

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
val_iter = torch.utils.data.DataLoader(mnist_val, batch_size=256, shuffle=False)

class MNet(nn.Module):
    def __init__(self):
        super(MNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        return self.fc(feature.view(img.shape[0], -1))

net = MNet()

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
print("train on", "cuda" if use_cuda else "cpu")

batch_size, num_epoch, lr = 256, 20, 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

loss = nn.CrossEntropyLoss()

net.train()
for epoch in range(num_epoch):
    train_acc, train_loss, n = 0.0, 0.0, 0
    for X, y in train_iter:
        if use_cuda:
            X = X.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        train_acc += (y_hat.argmax(dim=1) == y).sum().cpu().item()
        n += y.shape[0]
    train_acc /= n

    with torch.no_grad():
        net.eval()
        eval_loss, eval_acc, n = 0.0, 0.0, 0
        for X,y in val_iter:
            if use_cuda:
                X = X.cuda()
                y = y.cuda()

            y_hat = net(X)
            eval_loss += loss(y_hat, y).cpu().item()
            eval_acc += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
        net.train()
    print("epoch:%d, train_loss:%.4f, train_acc:%.4f, eval_loss:%.4f, eval_acc:%.4f" % (epoch + 1, train_loss, train_acc, eval_loss, eval_acc / n))





