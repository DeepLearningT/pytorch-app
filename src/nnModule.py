# -*- coding:utf-8 -*-
from torch.autograd import Variable

__author__ = 'tonye'

import torch.nn as nn
import torch.nn.functional as F
import torch as t

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层输入通道为1  输出通道为6 卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层， y = wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # reshape '-1' 表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 创建网络
net = Net()
print(net)

# 读取参数
params = list(net.parameters())
print(len(params))
for name, parameters in net.named_parameters():
    print(name, ":", parameters.size())


input = Variable(t.randn(1, 1, 32, 32))
#out = net(input)
# print out.size()

# # 所有参数的梯度归零
#net.zero_grad()
# # 反向传播
#out.backward(Variable(t.ones(1, 10)))
# print input.grad

# 计算交叉商损失
print t.arange(0, 10)
output = net(input)
target = Variable(t.arange(0, 10).float())
criterion = nn.MSELoss()
loss = criterion(output, target)
print loss
print 'grad_fn:', loss.grad_fn


# 运行.backword 观察调用之前和调用之后的grad
net.zero_grad()  # 把net中所有课学习参数的梯度清零
print '反向传播之前conv1.bias的梯度:'
print net.conv1.bias.grad
loss.backward()
print '反向传播之后conv1.bias的梯度:'
print net.conv1.bias.grad



import torch.optim as optim
# 新建一个优化器，指定要调整的参数和优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练过程中
# 先梯度清零（与net.zero_grad()效果一样)
optimizer.zero_grad()

# 计算损失
output = net(input)
loss = criterion(output, target)

# 反向传播
loss.backward()

# 更新参数
optimizer.step()

print '反向传播之后conv1.bias的梯度:'
print optimizer





