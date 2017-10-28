import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

# input is of size nBatch x nClasses = 3 x 5
output = Variable(torch.randn(3, 5), requires_grad=True)

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-4)
target = Variable(torch.LongTensor([1, 0, 4]))

l = loss(output, target)
l.backward()

print(output.data, target.data, l.data)
print(output.size(), target.size(), l.size())


# Cross entropy example
import numpy as np
Y = np.array([0, 1, 0])
Y_pred1 = np.array([0.1, 0.8, 0.1])
Y_pred2 = np.array([0.8, 0.1, 0.1])
print("loss1 = ", np.sum(-Y * np.log(Y_pred1)))
print("loss2 = ", np.sum(-Y * np.log(Y_pred2)))
