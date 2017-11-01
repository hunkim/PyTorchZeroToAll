import torch
import torch.nn as nn
from torch.autograd import Variable

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

# (num_layers * num_directions, batch, hidden_size)
# (batch, num_layers * num_directions, hidden_size) for batch_first=True
hidden = (Variable(torch.randn(1, 1, 2)))

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
inputs = Variable(torch.Tensor([[h, e, l, l, o]]))
print("input size", inputs.size())

for one in inputs[0]:
    one = one.view(1, 1, -1)
    # Input: (batch, seq_len, input_size) when batch_first=True
    out, hidden = cell(one, hidden)
    print(out.size())

# We can do the whole at once
# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
out, hidden = cell(inputs, hidden)
print("out size", out.size())


# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
# rank = (3, 5, 4)
inputs = Variable(torch.Tensor([[h, e, l, l, o],
                                [e, o, l, l, l],
                                [l, l, e, e, l]]))
print("input size", inputs.size())  # input size torch.Size([3, 5, 4])

# Propagate input through RNN
# Input: (batch, seq_len, input_size) when batch_first=True
# B x S x I
out, hidden = cell(inputs, hidden)
print("out size", out.size())  # out size torch.Size([3, 5, 2])
