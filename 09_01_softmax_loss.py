import torch
import torch.nn as nn

# Cross entropy example
import numpy as np
# One hot
# 0: 1 0 0
# 1: 0 1 0
# 2: 0 0 1
# Input(Y) is one hot.
Y = np.array([1, 0, 0])

Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print("loss1 = ", np.sum(-Y * np.log(Y_pred1)))
print("loss2 = ", np.sum(-Y * np.log(Y_pred2)))

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input(Y) is class, not one-hot
Y = torch.tensor([0], requires_grad=False)

# input is of size nBatch x nClasses = 1 x 4
# Y_pred are logits (not softmax)
Y_pred1 = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred2 = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("PyTorch Loss1 = ", l1.data, "\nPyTorch Loss2=", l2.data)

print("Y_pred1=", torch.max(Y_pred1.data, 1)[1])
print("Y_pred2=", torch.max(Y_pred2.data, 1)[1])

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input(Y) is class, not one-hot
Y = torch.tensor([2, 0, 1], requires_grad=False)

# input is of size nBatch x nClasses = 2 x 4
# Y_pred are logits (not softmax)
Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])


Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1 = ", l1.data, "\nBatch Loss2=", l2.data)
