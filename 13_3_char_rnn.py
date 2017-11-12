# https://github.com/spro/practical-pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from text_loader import TextDataset

hidden_size = 100
n_layers = 3
batch_size = 1
n_epochs = 100
n_characters = 128  # ASCII


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    # This runs this one step at a time
    # It's extremely slow, and please do not use in practice.
    # We need to use (1) batch and (2) data parallelism
    def forward(self, input, hidden):
        embed = self.embedding(input.view(1, -1))  # S(=1) x I
        embed = embed.view(1, 1, -1)  # S(=1) x B(=1) x I (embedding size)
        output, hidden = self.gru(embed, hidden)
        output = self.linear(output.view(1, -1))  # S(=1) x I
        return output, hidden

    def init_hidden(self):
        if torch.cuda.is_available():
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
        else:
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size)

        return Variable(hidden)


def str2tensor(string):
    tensor = [ord(c) for c in string]
    tensor = torch.LongTensor(tensor)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return Variable(tensor)


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = str2tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = chr(top_i)
        predicted += predicted_char
        inp = str2tensor(predicted_char)

    return predicted

# Train for a given src and target
# It feeds single string to demonstrate seq2seq
# It's extremely slow, and we need to use (1) batch and (2) data parallelism
# http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html.


def train_teacher_forching(line):
    input = str2tensor(line[:-1])
    target = str2tensor(line[1:])

    hidden = decoder.init_hidden()
    loss = 0

    for c in range(len(input)):
        output, hidden = decoder(input[c], hidden)
        loss += criterion(output, target[c])

    decoder.zero_grad()
    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / len(input)


def train(line):
    input = str2tensor(line[:-1])
    target = str2tensor(line[1:])

    hidden = decoder.init_hidden()
    decoder_in = input[0]
    loss = 0

    for c in range(len(input)):
        output, hidden = decoder(decoder_in, hidden)
        loss += criterion(output, target[c])
        decoder_in = output.max(1)[1]

    decoder.zero_grad()
    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / len(input)

if __name__ == '__main__':

    decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
    if torch.cuda.is_available():
        decoder.cuda()

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(dataset=TextDataset(),
                              batch_size=batch_size,
                              shuffle=True)

    print("Training for %d epochs..." % n_epochs)
    for epoch in range(1, n_epochs + 1):
        for i, (lines, _) in enumerate(train_loader):
            loss = train(lines[0])  # Batch size is 1

            if i % 100 == 0:
                print('[(%d %d%%) loss: %.4f]' %
                      (epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, 'Wh', 100), '\n')
