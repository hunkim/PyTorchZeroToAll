# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from text_loader import TextDataset
import seq2seq_modes as sm
import argparse


SOS_token = chr(0)
EOS_token = 1

HIDDEN_SIZE = 100
N_LAYERS = 3
BATCH_SIZE = 32
N_EPOCHES = 10
N_CHARS = 128  # ASCII


# Sting to char tensor
def str2tensor(str, eos=False):
    tensor = [ord(c) for c in str]
    if eos:
        tensor.append(EOS_token)
    tensor = torch.LongTensor(tensor)

    # Do before wrapping with variable
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return Variable(tensor)


# Simple test to show how our train works
def test():
    encoder_hidden = encoder.init_hidden()
    word_input = str2tensor('hello')
    encoder_outputs, encoder_hidden = encoder(word_input, encoder_hidden)
    print(encoder_outputs)

    decoder_hidden = encoder_hidden

    word_target = str2tensor('pytorch')
    for c in range(len(word_target)):
        decoder_output, decoder_hidden = decoder(
            word_target[c], decoder_hidden)
        print(decoder_output.size(), decoder_hidden.size())


# Train for a given src and target
# It feeds single string to demonstrate seq2seq
# It's extremely slow, and we need to use (1) batch and (2) data parallelism
# http://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html.
def train(src, target):
    src_var = str2tensor(src)
    target_var = str2tensor(target, eos=True)  # Add the EOS token

    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(src_var, encoder_hidden)

    hidden = encoder_hidden
    loss = 0

    for c in range(len(target_var)):
        # First, we feed SOS
        # Others, we use teacher forcing
        token = target_var[c - 1] if c else str2tensor(SOS_token)
        output, hidden = decoder(token, hidden)
        loss += criterion(output, target_var[c])

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / len(target_var)


# Translate the given input
def translate(enc_input='thisissungkim.iloveyou.', predict_len=100, temperature=0.9):
    input_var = str2tensor(enc_input)
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)

    hidden = encoder_hidden

    predicted = ''
    dec_input = str2tensor(SOS_token)
    for c in range(predict_len):
        output, hidden = decoder(dec_input, hidden)

        # Sample from the network as a multi nominal distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Stop at the EOS
        if top_i is EOS_token:
            break

        predicted_char = chr(top_i)
        predicted += predicted_char

        dec_input = str2tensor(predicted_char)

    return enc_input, predicted


encoder = sm.EncoderRNN(N_CHARS, HIDDEN_SIZE, N_LAYERS)
decoder = sm.DecoderRNN(HIDDEN_SIZE, N_CHARS, N_LAYERS)

if torch.cuda.is_available():
    decoder.cuda()
    encoder.cuda()
print(encoder, decoder)
test()

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


train_loader = DataLoader(dataset=TextDataset(),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2)

print("Training for %d epochs..." % N_EPOCHES)
for epoch in range(1, N_EPOCHES + 1):
    # Get srcs and targets from data loader
    for i, (srcs, targets) in enumerate(train_loader):
        for src, target in zip(srcs, targets):
            train_loss = train(src, target)

        print('[(%d %d%%) %.4f]' %
              (epoch, epoch / N_EPOCHES * 100, train_loss))
        print(translate(srcs[0]), '\n')
        print(translate(), '\n')
