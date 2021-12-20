##############################################################################
# Training script for training CharRNN model to generate next character with #
# current character. Same trainer script in GoogleColab can be accessed with #
# this link -                                                                #
# https://colab.research.google.com/drive/1PquhW_ziRAEvRwPEDd-gE55-6r6L9Pde  #
##############################################################################

import string
import random
import requests

from tqdm import tqdm
import torch
import torch.nn as nn

url = 'https://raw.githubusercontent.com/cos495/code/master/shakespeare.txt'
data = requests.get(url).text

all_characters = string.printable
n_characters = len(all_characters)

hidden_size = 300
learning_rate = 0.01
n_layers = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_len = 100
batch_size = 128
epochs = 1
prediction_seq_length = 70
print_every = 1

print(f'Which device: {device}')
print(f'Number of Characters: {n_characters}')
print(all_characters)


def get_data():
    min_index = 0
    max_index = len(data) - seq_len - 2
    inputs = []
    targets = []
    for b in range(batch_size):
        start = random.randint(min_index, max_index)
        end = start + seq_len
        inputs.append(str2int(string_data=data[start:end]))
        targets.append(str2int(string_data=data[start + 1: end + 1]))
    inputs = torch.tensor(inputs).to(dtype=torch.long, device=device)
    targets = torch.tensor(targets).to(dtype=torch.long, device=device)
    return inputs, targets


def str2int(string_data):
    return [all_characters.index(c) for c in string_data]


def int2str(int_data):
    return ''.join([all_characters[i] for i in int_data])


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        inputs = self.encoder(inputs.view(1, -1))
        output, hidden = self.gru(inputs, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=device)


generator = CharRNN(
    input_size=n_characters,
    hidden_size=hidden_size,
    output_size=n_characters,
    n_layers=n_layers,
)

generator.to(device=device)
optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(1, epochs + 1)):
    inputs, targets = get_data()
    hidden = generator.init_hidden(batch_size)
    optimizer.zero_grad()
    loss = 0
    for j in range(seq_len):
        output, hidden = generator(inputs[:, j], hidden)
        loss += loss_fn(output.squeeze(), targets[:, j])
    loss.backward()
    optimizer.step()
    if epoch % print_every == 0:
        print('Loss: ', loss.item() / seq_len)
        hidden = generator.init_hidden(batch_size=1)
        primer = torch.tensor(str2int('RedisAI')).to(dtype=torch.long, device=device)
        for p in primer:
            _, hidden = generator(p, hidden)
        prob = []
        limit = prediction_seq_length
        while limit:
            output, hidden = generator(p, hidden)
            output_dist = output.squeeze().div(0.8).exp()
            p = torch.multinomial(output_dist, 1)[0]
            prob.append(p.item())
            limit -= 1
        print(int2str(prob))

generator.eval()
traced_generator = torch.jit.trace(generator, (p, hidden))
torch.jit.save(traced_generator, 'charrnn_model.pt')
