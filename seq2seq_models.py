# Original code from
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
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        super(EncoderRNN, self).__init__()

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

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # input shape: S(=1) x B (=1) x I (input size)
        # Note: we run this one step at a time. (Sequence size = 1)
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        # No need softmax, since we are using CrossEntropyLoss
        return output, hidden

    def init_hidden(self):
        # (num_layers * num_directions, batch, hidden_size)
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Linear for attention
        self.attn = nn.Linear(hidden_size, hidden_size)

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,
                          n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_hiddens):
        # Note: we run this one step (S=1) at a time
        # Get the embedding of the current input word (last output word)
        rnn_input = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x I
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs
        attn_weights = self.get_att_weight(
            rnn_output.squeeze(0), encoder_hiddens)
        context = attn_weights.bmm(
            encoder_hiddens.transpose(0, 1))  # B x S(=1) x I

        # Final output layer (next word prediction) using the RNN hidden state
        # and context vector
        rnn_output = rnn_output.squeeze(0)  # S(=1) x B x I -> B x I
        context = context.squeeze(1)  # B x S(=1) x I -> B x I
        output = self.out(torch.cat((rnn_output, context), 1))

        # Return final output, hidden state, and attention weights (for
        # visualization)
        return output, hidden, attn_weights

    def get_att_weight(self, hidden, encoder_hiddens):
        seq_len = len(encoder_hiddens)

        # Create variable to store attention energies
        attn_scores = cuda_variable(torch.zeros(seq_len))  # B x 1 x S

        # Calculate energies for each encoder hidden
        for i in range(seq_len):
            attn_scores[i] = self.get_att_score(hidden, encoder_hiddens[i])

        # Normalize scores to weights in range 0 to 1,
        # resize to 1 x 1 x seq_len
        # print("att_scores", attn_scores.size())
        return F.softmax(attn_scores).view(1, 1, -1)

    # score = h^T W h^e = h dot (W h^e)
    # TODO: We need to implement different score models
    def get_att_score(self, hidden, encoder_hidden):
        score = self.attn(encoder_hidden)
        return torch.dot(hidden.view(-1), score.view(-1))
