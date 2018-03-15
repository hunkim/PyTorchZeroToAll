"""Autogradient Example."""
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)  # Any random value


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# plotting purposes
w_history = []
loss_history = []

# Before training
print("predict (before training)", 4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        lossvalue = loss(x_val, y_val)
        lossvalue.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        # log values
        w_history.append(w.data[0])
        loss_history.append(lossvalue.data[0])
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, lossvalue.data[0])

# After training
print("predict (after training)", 4, forward(4).data[0])

# Plot Graphs
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
style = ('s', '8', 'X', 'p', 'd', 'D', '*')

# Plot weight vs Loss
plt.subplot(2, 1, 1)
plt.plot(w_history, loss_history, marker=style[1], color=colors[1],
         markersize=5)
plt.ylabel('Loss')
plt.xlabel('w')

# Plot iterations vs weight
plt.subplot(2, 1, 2)
plt.plot(range(1, len(w_history) + 1), w_history, marker=style[1],
         color=colors[1], markersize=5)
plt.ylabel('w')
plt.xlabel('Iteration')

plt.show()
