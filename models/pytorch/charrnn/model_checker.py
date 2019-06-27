import torch
import string

charrnn = torch.jit.load('CharRNN_pipeline.pt')

all_characters = string.printable
n_characters = len(all_characters)
hidden_size = 300
n_layers = 2
batch_size = 1
prediction_seq_length = 200


def int2str(int_data):
    return ''.join([all_characters[i] for i in int_data])


primer = torch.tensor([5]).to(dtype=torch.long)
hidden = torch.zeros(n_layers, batch_size, hidden_size)
output = charrnn(primer, hidden)
print(int2str(output))
