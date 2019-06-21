import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]] )


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
#criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
    
    #print(y_pred,'\n', y_data)
    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training
hour_var = torch.tensor([[1.0]])
print("predict 1 hour ", 1.0, model(hour_var).data[0][0] > 0.5)
hour_var = torch.Tensor([[7.0]])
print("predict 7 hours", 7.0, model(hour_var).data[0][0] > 0.5)
