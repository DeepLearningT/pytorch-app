# -*- coding:utf-8 -*-


__author__ = 'tonye'

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

# 第一次运行程序torchvision会自动下载CIFAR-10数据集
# 如果已经下载有CIFAR-10, 可以通过root参数指定


# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='../data/',
    train=True,
    download=True,
    transform=transform
)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=10
)

# 测试集
testset = tv.datasets.CIFAR10(
    root='../data/',
    train=False,
    download=True,
    transform=transform
)

testLoader = t.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=10
)

classes = (
    'place', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'shop', 'truck'
)

(data, label) = trainset[100]
print classes[label]

# (data + 1) / 2是为了还原被归一化的数据
#show((data + 1) / 2).resize((100, 100)).show()




dataiter = iter(trainloader)
images, labels = dataiter.next()
print ' '.join('%11s'%classes[labels[j]] for j in range(4))
#show(tv.utils.make_grid((images + 1)/2)).resize((400, 100)).show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print net

from torch import optim
criterion = nn.CrossEntropyLoss()  # 交叉商损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 输入数据
        inputs, labels = data
        if False:
            net.cuda()
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)  # Variable是自动微分Autograd中的核心类

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        running_loss += loss.data
        if i % 200 == 199:  # 每20个batch打印一次训练状态
            print '[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000)
            running_loss = 0.0

    print 'Finished Training'

# 测试一个batch
dataiter = iter(testLoader)
images, labels = dataiter.next()  # 一个batch返回4张图片
print '实际的label: ', ' '.join('%08s'%classes[labels[j]] for j in range(4))
#show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100)).show()

# 计算图片在每个类别的分数
outputs = net(Variable(images))
# 得分最高的那个类
_, predicted = t.max(outputs.data, 1)

print '预测结果：', ' '.join('%5s'%classes[predicted[j]] for j in range(4))


# 测试10000张
correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数
for data in testLoader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print '10000张测试集中的准确率为: %d %%' % (100 * correct / total)



