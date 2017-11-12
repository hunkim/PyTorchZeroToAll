# Original source from
# https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
# torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools


def flatten(l):
    return list(itertools.chain.from_iterable(l))

seqs = ['ghatmasala', 'nicela', 'chutpakodas']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(list(set(flatten(seqs))))

# make model
embedding_size = 3
embed = nn.Embedding(len(vocab), embedding_size)
lstm = nn.LSTM(embedding_size, 5)

vectorized_seqs = [[vocab.index(tok) for tok in seq]for seq in seqs]
print("vectorized_seqs", vectorized_seqs)

print([x for x in map(len, vectorized_seqs)])
# get the length of each seq in your batch
seq_lengths = torch.LongTensor([x for x in map(len, vectorized_seqs)])

# dump padding everywhere, and place seqs on the left.
# NOTE: you only need a tensor as big as your longest sequence
seq_tensor = Variable(torch.zeros(
    (len(vectorized_seqs), seq_lengths.max()))).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

print("seq_tensor", seq_tensor)

# SORT YOUR TENSORS BY LENGTH!
seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]

print("seq_tensor after sorting", seq_tensor)

# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors
seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)
print("seq_tensor after transposing", seq_tensor.size(), seq_tensor.data)

# embed your sequences
embeded_seq_tensor = embed(seq_tensor)
print("seq_tensor after embeding", embeded_seq_tensor.size(), seq_tensor.data)

# pack them up nicely
packed_input = pack_padded_sequence(
    embeded_seq_tensor, seq_lengths.cpu().numpy())

# throw them through your LSTM (remember to give batch_first=True here if
# you packed with it)
packed_output, (ht, ct) = lstm(packed_input)

# unpack your output if required
output, _ = pad_packed_sequence(packed_output)
print("Lstm output", output.size(), output.data)

# Or if you just want the final hidden state?
print("Last output", ht[-1].size(), ht[-1].data)
