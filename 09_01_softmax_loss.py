import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# http://pytorch.org/docs/master/nn.html#nllloss
logsm = nn.LogSoftmax()
loss = nn.NLLLoss()

# input is of size nBatch x nClasses = 3 x 5
input = Variable(torch.randn(3, 5), requires_grad=True)
logsm_out = logsm(input)

# target is of size nBatch
# each element in target has to have 0 <= value < nclasses
target = Variable(torch.LongTensor([1, 0, 4]))

l = loss((logsm_out), target)
l.backward()

print(input.size(), target.size(), l.size())
