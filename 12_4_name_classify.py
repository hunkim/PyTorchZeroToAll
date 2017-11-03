# https://github.com/spro/practical-pytorch
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from name_dataset import NameDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HIDDEN_SIZE = 100
N_LAYERS = 1
BATCH_SIZE = 32
N_EPOCHS = 20

test_dataset = NameDataset(is_test_set=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)


train_dataset = NameDataset(is_test_set=False)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

n_countries = len(train_dataset.get_countries())
N_CHARS = 128  # ASCII


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def cuda_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


# Sting to char tensor
def pad_sequences(vectorized_seqs, seq_lengths, countries):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    target = countries2tensor(countries)
    target = target[perm_idx]

    return cuda_variable(seq_tensor), \
        cuda_variable(seq_lengths), \
        cuda_variable(target)


# vectorized_seqs, seq_lengths
def make_variables(names, countries):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = [sl[1] for sl in sequence_and_length]
    return pad_sequences(vectorized_seqs, torch.LongTensor(seq_lengths), countries)


def str2ascii_arr(msg):
    arr = [ord(c) for c in msg]
    return arr, len(arr)


def countries2tensor(countries):
    country_ids = [train_dataset.get_country_id(
        country) for country in countries]
    return torch.LongTensor(country_ids)


class RNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: S x B (input size)
        batch_size = len(input[1])

        # Get hidden
        hidden = self._init_hidden(batch_size)

        # input shape: S x B (input size)
        embeded = self.embedding(input)  # S x B -> S x B x I (embedding size)

        gru_input = embeded
        # FIXME: is this a right way? It makes training and testing slower
        # With pack: [1m 0s (20 95%) 0.0596]
        # Without pack: [0m 45s (20 95%) 0.0901]
        # pack them up nicely
        # gru_input = pack_padded_sequence(embeded, seq_lengths.cpu().numpy())

        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()

        output, hidden = self.gru(gru_input, hidden)
        fc_output = self.fc(hidden[-1])  # Use the last layer
        print("Final output size", fc_output.size())
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return cuda_variable(hidden)


# Train for a given src and target
# It feeds single string to demonstrate seq2seq
# It's extremely slow, and we need to use (1) batch and (2) data parallelism
# http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html.
def train(names, countries):
    input, seq_lengths, target = make_variables(names, countries)

    # transpose to make S(sequence) x B (batch)
    output = classifier(input.t(), seq_lengths)

    # FIXME: output size is Batch*n_GPUs * Inputsize
    print("output size", output.size())
    loss = criterion(output, target)

    classifier.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0]


def test():
    print("evaluating ...")
    correct = 0

    for i, (names, countries) in enumerate(test_loader):
        input, seq_lengths, target = make_variables(names, countries)

        # transpose to make S(sequence) x B (batch)
        output = classifier(input.t(), seq_lengths)
        print(i, output.size())
        print(target.size())

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, n_countries, N_LAYERS)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier)

    if torch.cuda.is_available():
        classifier.cuda()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        loss = 0

        for i, (names, countries) in enumerate(train_loader):
            loss += train(names, countries)  # Batch size is 1

            if i % 100 == 0:
                print('[%s (%d %d%%) %.4f]' %
                      (time_since(start), epoch, i * BATCH_SIZE * 100 / len(train_loader.dataset), loss / (i + 1)))

        # Testing
        test()
