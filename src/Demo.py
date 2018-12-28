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
