# -*- coding: utf-8 -*-


import torch
import numpy as np
import matplotlib.pyplot as plt

w = torch.tensor( 3.0, requires_grad=True)
b = torch.tensor( 1.0, requires_grad=True)

"""Def the forward function:"""

def forward(x):
  y = w*x + b
  return y

x = torch.tensor([[2], [4], [7]])
forward(x)

"""Now NN:
**Create linear Model**
"""

from torch.nn import Linear

torch.manual_seed(1)
model = Linear(in_features=1, out_features=1) # for every input, there is a single output
print(model.bias, model.weight)

x = torch.tensor([2.0])
print(model(x))

class LR(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = self.linear(x)
    return pred

torch.manual_seed(1)
model = LR(1, 1)
print(list(model.parameters()))

x = torch.Tensor([[1.0], [2.0]])
print(model.forward(x))



"""Linear Regression ( Custome Modules):"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""make dataset:"""

x = torch.randn(100,1) # 100 points in 1 colomn
#print(x)
y = x
plt.plot(x.numpy(), y.numpy(), 'o')

"""Add noise:"""

x = torch.randn(100,1)*10 # 100 points in 1 colomn
#print(x)
y = x + 3*torch.randn(100,1)
plt.plot(x.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')

class LR(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.linear = nn.Linear(input_size, output_size)
  def forward(self, x):
    pred = self.linear(x)
    return pred

torch.manual_seed(1)
model = LR(1,1)

[w, b] = model.parameters()
print(x,b)