import json

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# TODO: `.to(device=device)` for all tensors


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc.num_words, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers,
            dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        t1 = outputs[:, :, :self.hidden_size]
        t2 = outputs[:, :, self.hidden_size:]
        outputs = t1 + t2
        # only first part of the hidden layer is required for decoder
        return outputs, hidden[:self.n_layers]


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_output):
        attn_energies = torch.sum(hidden * encoder_output, dim=2)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size,
            output_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(voc.num_words, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers,
            dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS"}
        self.num_words = len(self.index2word)


voc = Voc(name=None)

with open('voc.json') as f:
    voc.__dict__ = json.load(f)

hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
MAX_LENGTH = 10

encoder = EncoderRNN(hidden_size, encoder_n_layers, dropout)
encoder_params = torch.load('weights/encoder.pth', map_location='cpu')
encoder.load_state_dict(encoder_params)
encoder.eval()
seq = torch.ones((MAX_LENGTH, 1), dtype=torch.long)
seq_length = torch.tensor([seq.size()[0]])
traced_encoder = torch.jit.trace(encoder, (seq, seq_length))

decoder = LuongAttnDecoderRNN(
    hidden_size, voc.num_words, decoder_n_layers, dropout)
decoder_params = torch.load('weights/decoder.pth', map_location='cpu')
decoder.load_state_dict(decoder_params)
decoder.eval()
test_encoder_outputs, test_encoder_hidden = traced_encoder(seq, seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
traced_decoder = torch.jit.trace(
    decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))


traced_encoder.save('encoder.pt')
traced_decoder.save('decoder.pt')
