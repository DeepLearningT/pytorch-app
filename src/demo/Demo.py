# -*- coding:utf-8 -*-
__author__ = 'tonye'

import torch as t

x = t.Tensor(5, 3)
print x

x = t.rand(5, 3)
print x

print x.size()


from torch.autograd import Variable

x = Variable(t.ones(2, 2), requires_grad=True)
print x

y = x.sum()
print y

y.backward()
print x.grad
y.backward()
print x.grad
y.backward()
print x.grad

x.grad.data.zero_()
#y.backward()
print x.grad


y = t.cos(x)
print(y)

b_size = x.size()
print b_size

print x.numel()

c = t.Tensor(b_size)
d = t.Tensor((2,3))
print c, d

print c.shape

print t.arange(1, 6, 2)
print t.linspace(1, 7, 3)

print t.randperm(5)

print t.eye(3, 3)

a = t.arange(0, 6)
print a.view(2, 3)
b = a.view(-1, 3)
print b

print b.unsqueeze(0).shape
print b.unsqueeze(1).shape
print b.unsqueeze(-2).shape

c = b.view(1, 1, 1, 2, 3)
print c
print c.squeeze(0).shape  # 第'0'维的1压缩
print c.squeeze().shape  # 把所有维度为 '1' 的压缩

a[1] = 100  # a和b共享内存，修改了a,b也变了
print b

print b.resize_(1, 3)
print b.resize_(3, 3)


a = t.randn(3, 4)
print a
print a[0]
print a[:, 0]
print a[:2, 0:2]  # 前两行前两列

# 注意两者的形状不同
print a[0:1, 0:2].shape
print a[0, 0:2].shape

print a > 1

# 这种方式选择结果与源tensor不共享内存
print a[a>1]
print a.masked_select(a>1)

# 将第0行和第1行转成LongTensor
print a[t.LongTensor([0, 1])].shape

t.set_default_tensor_type('torch.IntTensor')
cc = t.Tensor(2, 3)
print cc.type()