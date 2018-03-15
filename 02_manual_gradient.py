"""Autogradient Example."""
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0    # a random guess of initial weight
lr = 0.01  # learning rate
num_epoches = 10
# our model forward pass


def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# compute gradient
# d_loss/d_w
# loss = (wx-y)^2
def gradient(x, y):
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)", 4, forward(4))

# plotting purposes
w_history = []
loss_history = []
# Training loop
for epoch in range(num_epoches):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - lr * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        loss_val = loss(x_val, y_val)
        # log values
        w_history.append(w)
        loss_history.append(loss_val)

    print("progress:", epoch, "w=", round(w, 2), "loss=", round(loss_val, 2))

# After training
print("predict (after training)", "4 hours", forward(4))

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
