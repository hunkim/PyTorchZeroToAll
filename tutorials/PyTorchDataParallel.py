"""
DataParallel
============

Authors: Sung Kim hunkim@gmail.com and Jenny Kang

If you have GPUs, it's very easy to use them in PyTorch. Just you put
the model on GPU:

.. code:: python

    model.gpu()

Then, you can copy all your tensors to GPU:

.. code:: python

    mytensor = my_tensor.gpu()

Please note that just calling ``mytensor.gpu()`` won't copy the tensor
to GPU. You need to assign it to a new tensor and use the tensor on GPU.

Furthermore, it's natural to execute your long-waiting forward, backward
propagations on multiple GPUs. Unfortunately, PyTorch won't do that
automatically for you. Not yet. (It will just use one GPU for you.)

However, running your operations on multiple GPUs is very easy. Just you
need to make your model dataparallelable using this.

.. code:: python

    model = nn.DataParallel(model)

That's it. If you want to know more, here we are!

"""


######################################################################
# Imports and parameters
# ----------------------
# 
# Let's import our favorite core PyTorch things and define some
# parameters.
# 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


######################################################################
# Dummy DataSet
# -------------
# 
# It's fun to play with dataloader. Let's make a dummy (random) one. Just
# need to implement the getitem!
# 

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, 100),
                         batch_size=batch_size, shuffle=True)


######################################################################
# Simple Model
# ------------
# 
# Then, we need a model to run. For DataParallel demo, let's make a simple
# one. Just get an input and do a linear operation, and output. However,
# you can make any model including CNN, RNN or even Capsule Net for
# ``DataParallel``.
# 
# Inside of the model, we just put a print statement to monitor the size
# of input and output tensors. Please pay attention to the batch part,
# rank 0 when they print out something.
# 

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(), 
              "output size", output.size())

        return output


######################################################################
# Create Model and DataParallel
# -----------------------------
# 
# Here is the core part. First, make a model instance, and check if you
# have multiple GPUs. (If you don't, I feel sorry for you.) If you have,
# just wrap our model using ``nn.DataParallel``. That's it. I know, it's
# hard to believe, but that's really it!
# 
# Then, finally put your model on GPU by ``model.gpu()``. It's simple and
# beautiful.
# 

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

if torch.cuda.is_available():
   model.cuda()


######################################################################
# Fun part
# --------
# 
# Now it's the fun part. Just get data from the dataloader and see the
# size of input and out tensors!
# 

for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)

    output = model(input_var)
    print("Outside: input size", input_var.size(),
          "output_size", output.size())


######################################################################
# Didn't you see?
# ---------------
# 
# Hmm, did you see something working here? It seems just batch 30 input
# and output 30. The model gets 30 and spits out 30. Nothing special.
# 
# BUT, Wait! This notebook (or yours) does not have GPUs. If you have
# GPUs, the execution looks like this, called DataParallel!
# 
# 2 GPUs
# ~~~~~~
# 
# .. code:: bash
# 
#     # on 2 GPUs
#     Let's use 2 GPUs!
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
# 
# 3 GPUs
# ~~~~~~
# 
# If you have 3 GPUs, you will see:
# 
# .. code:: bash
# 
#     Let's use 3 GPUs!
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
# 
# Amazing 8 GPUs
# ~~~~~~~~~~~~~~
# 
# If you have 8, it's amazing, and you will see this:
# 
# .. code:: bash
# 
#     Let's use 8 GPUs!
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
# 


######################################################################
# Summary
# -------
# 
# DataParallel splits your data automatically, and send job orders to
# multiple models on different GPUs using the data. After each model
# finishes their job, DataParallel collects and merges the results for
# you. It's really awesome!
# 
# For more information, please check out
# http://pytorch.org/tutorials/beginner/former\_torchies/parallelism\_tutorial.html
# and slides at http://bit.ly/PyTorchZeroAll.
# 