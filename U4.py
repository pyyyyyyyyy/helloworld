#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
from torch.nn import init


# In[16]:


### 构造网络结构
#1 通过直接继承Module构造：
class MLP(nn.Module):
    def __init__(self,**kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(5,784)
net = MLP()
print(net)
print(net(X))


# In[17]:


### 构造网络结构
#2 通过继承Module子类Sequential构造：

net = nn.Sequential(nn.Linear(784, 256) ,nn.ReLU(), nn.Linear(256, 10))
print(net)
X = torch.rand(5,784)
print(net(X))


# In[18]:


### 构造网络结构
#3 通过继承Module子类ModuleDict构造：
# 作为字典操作

net = nn.ModuleDict({'hidden':nn.Linear(784, 256),
                    'act':nn.ReLU(),
                    })
net['output'] = nn.Linear(256, 10)
print(net)
print(net['act'])
print(net.output)


# In[19]:


### 构造网络结构
#4 通过继承Module子类ModuleList构造：
# 作为列表，可使用append、extend等操作

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net)
print(net[-1])


# In[ ]:





# In[40]:


net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
# pytorch已默认进行初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()


# In[41]:


print(X,Y)


# In[42]:


# 通过继承Module类中paramters()或named_paramters()方法，来访问网络中所有参数（以迭代器形式返回）。后者除了返回参数tensor还会返回名字
print(type(net.named_parameters()))

for name, param in net.named_parameters():
    print(name, param.size())

# param是一个tensor，data(),size(),grad()
# 可见返回的名字前自动带了层数前缀


# In[43]:


# 访问单层参数
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))

# 因为是单层访问，所以名字前方没有数字前缀


# In[44]:


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
        
    def forward(self,x):
        pass

net0 = MyModel()
print(net0)
for name, param in net0.named_parameters():
    print(name)
    
# weight1使用了nn.Parameter(),在参数列表中
# weight2不在参数列表中


# In[45]:


# Parameter本质是tensor,tensor有的属性它都有，例如可以通过data()访问参数数值，通过grad()访问参数梯度
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)
# 反向传播前梯度为None

Y.backward()
print(weight_0.grad)


# In[46]:


### 初始化模型参数
# 使用pytorch的init中自带的多种预设的初始化方法来初始化权重（torch.nn.init.normal_())
for name, param in net.named_parameters():
    if 'weight' in name:
        # 对权重进行随机正态分布
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)


# In[47]:


# 使用常数进行初始化(torch.nn.init.constant_())
for name, param in net.named_parameters():
    if 'bias' in name:
        # 对偏置进行常数初始化
        init.constant_(param, val=0)
        print(name, param.data)


# In[48]:


for name, param in net.named_parameters():
    print(name,param)


# In[51]:


# 使用自定义的方法初始化权重。
# 参考init中预定义的初始化方法，可以看到就是一个改变tensor数值的函数，且这个过程中不记录梯度。

def init_weight_(tensor):
    # 初始化过程中不记录梯度
    with torch.no_grad():
        # uniform_(x, y)表示在[x, y]中随机抽样
        tensor.uniform_(-10, 10)
        # 表示有一半的概率为0，另一半的概率初始化为[-10，-5],[5, 10]之间均匀分布的随机数
        tensor *= (tensor.abs() >= 5).float()
        
for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)


# In[55]:


# 改变bias值（param.data），同时不影响梯度

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)


# In[ ]:





# In[59]:


# 共享模型参数
# 如果传入Sequential的是同一个Module实例，则他们的参数共享
linear = nn.Linear(3, 3, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=2)
    print(name, param.size(), param.data)
    

# 网络的两层对应同一个对象
print(id(net[0]) == id(net[1]))


# In[63]:


X = torch.ones(5, 3)
y = net(X).sum()
print(X,y)

# 因为是同一个对象，所以共享的参数梯度是累加的
y.backward()
print(net[0].weight.grad)


# In[ ]:





# In[74]:


# 自定义层
# 定义一个不含模型参数的自定义层

class CentreLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CentreLayer,self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()
    
layer = CentreLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))


# In[70]:


net = nn.Sequential(nn.Linear(8, 128), CentreLayer())

y = net(torch.rand(4, 8))

y.mean().item()


# In[92]:


# 定义一个含模型参数的自定义层（参数应该定义成nn.Parameter类型，可以自动被识别成模型参数）
# 参数可使用ParameterList.(append,extend)

class MyListDense(nn.Module):
    def __init__(self, **kwargs):
        super(MyListDense, self).__init__(**kwargs)
        # ParameterList是一个列表，每个元素均为nn.Parameter类型
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(4,4)) for i in range(3)]
        )
        self.params.append(nn.Parameter(torch.ones(4,4)))
        
    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

net = MyListDense()
print(net)


# In[93]:


# 定义一个含模型参数的自定义层（参数应该定义成nn.Parameter类型，可以自动被识别成模型参数）
# 参数可使用ParameterDict.(update,keys)

class MyDictDense(nn.Module):
    def __init__(self, **kwargs):
        super(MyDictDense, self).__init__(**kwargs)
        self.params = nn.ParameterDict({
            'weight0':nn.Parameter(torch.randn(4, 4)),
            'weight1':nn.Parameter(torch.randn(4, 1))                    
        })
        self.params.update({'weight2':nn.Parameter(torch.randn(4, 2))})
        
    def forward(self, x, choice= 'weight0'):
        return torch.mm(x, self.params[choice])
    
net = MyDictDense()
print(net)

x = torch.ones(1,4)
print(net(x, 'weight0'))
print(net(x, 'weight1'))
print(net(x, 'weight2'))


# In[94]:


net = nn.Sequential(MyListDense(), MyDenseDict())
print(net)
print(net(x))


# In[ ]:





# In[95]:


# 读取和存储
# 使用save和load读取和保存tensor

x = torch.ones(3)
torch.save(x, 'x.pt')


# In[97]:


x2 = torch.load('x.pt')
x2


# In[99]:


# 保存一个tensor列表

y = torch.zeros(4)
torch.save([x,y],'xy_list.pt')
xy = torch.load('xy_list.pt')
xy


# In[100]:


# 保存一个tensor字典

torch.save({'x':x,'y':y},'xy_dict.pt')
xy = torch.load('xy_dict.pt')
xy


# In[106]:


# 读写模型

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(3,5)
        self.act = nn.ReLU()
        self.output = nn.Linear(5,2)
        
    def forward(self ,x):
        return output(act(x))
    
net = MLP()
net.state_dict()

# state_dict()返回一个从参数名称映射到参数tensor的字典，只有可学习的对象才会出现在字典中


# In[109]:


optimizer = torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
optimizer.state_dict()


# In[ ]:


# 保存和加载模型

# 保存
# torch.save(model.state_dict(), PATH)

# 加载
# model = themodelclass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))


# In[ ]:





# In[ ]:


# GPU计算

# .cuda()可以将CPU上的tensor转换到GPU上

# 指定设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

