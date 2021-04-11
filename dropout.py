import torch
import torch.nn
from presentation import *

num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256

batch_size, num_epochs, lr = 256, 5, 0.1
train_iter, test_iter = load_data_fashion_mnist(batch_size)
drop_prob1, drop_prob2 = 0.2, 0.5

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hidden1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hidden1, num_hidden2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hidden2, num_outputs)
)
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr)

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)