# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

MAX_LENGTH = 100

SOS_token = chr(0)
EOS_token = 1

# Helper function to create Variable based on
# the cuda availability


def cuda_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


# Sting to char tensor
def str2tensor(msg, eos=False):
    tensor = [ord(c) for c in msg]
    if eos:
        tensor.append(EOS_token)

    return cuda_variable(torch.LongTensor(tensor))


# To demonstrate seq2seq, We don't handle batch in the code,
# and our encoder runs this one step at a time
# It's extremely slow, and please do not use in practice.
# We need to use (1) batch and (2) data parallelism
# http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html.

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        # input shape: S x B (=1) x I (input size)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        # (num_layers * num_directions, batch, hidden_size)
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        # input shape: S(=1) x B (=1) x I (input size)
        # Note: we run this one step at a time. (Sequence size = 1)
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        # (num_layers * num_directions, batch, hidden_size)
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))


class Attn(nn.Module):

    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = cuda_variable(torch.zeros(seq_len))  # B x 1 x S

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x
        # seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.other.view(-1), energy.view(-1))
            return energy


class AttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size,
                          n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step (S=1) at a time
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(
            word_input).view(1, 1, -1)  # S=1 x B x I

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # B x S(=1) x I

        # Final output layer (next word prediction) using the RNN hidden state
        # and context vector
        rnn_output = rnn_output.squeeze(0)  # S(=1) x B x I -> B x I
        context = context.squeeze(1)  # B x S(=1) x I -> B x I
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for
        # visualization)
        return output, context, hidden, attn_weights
