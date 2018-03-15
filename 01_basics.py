"""Basics."""
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])
N = x_data.size + 1

# our model for the forward pass


def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print("\t", x_val, y_val, y_pred_val, loss_val)
    print("MSE=", l_sum / N)
    w_list.append(w)
    mse_list.append(l_sum / N)

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
style = ('s', '8', 'X', 'p', 'd', 'D', '*')
plt.plot(w_list, mse_list, marker=style[1], color=colors[1], markersize=5)
plt.plot(w_list, [x + 2 for x in mse_list], marker=style[0], color=colors[0],
         markersize=5)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
