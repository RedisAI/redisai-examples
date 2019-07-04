import json
import torch


PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# TODO: `.to(device=device)` for all tensors


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
        self.num_words = 3  # Count SOS, EOS, PAD


voc = Voc(name=None)

with open('voc.json') as f:
    voc.__dict__ = json.load(f)


encoder = torch.jit.load('encoder.pt')
decoder = torch.jit.load('decoder.pt')


def run():
    with torch.no_grad():
        SOS_token = 1
        max_length = 10
        indexes_batch = [[787, 572, 2]]  # "hello sir + EOS"
        lengths = torch.tensor([3])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # d_hidden = e_hidden
        e_output, d_hidden = encoder(input_batch, lengths)
        d_input = torch.ones(1, 1, dtype=torch.long)
        d_input *= SOS_token
        all_tokens = torch.zeros([0], dtype=torch.long)
        while max_length > 0:
            max_length -= 1
            d_output, d_hidden = decoder(d_input, d_hidden, e_output)
            _, d_input = torch.max(d_output, dim=1)
            if d_input.item() == EOS_token:
                break
            all_tokens = torch.cat((all_tokens, d_input), dim=0)
            d_input = torch.unsqueeze(d_input, 0)
        output_words = [voc.index2word[str(token.item())] for token in all_tokens]
        out_string = []
        for x in output_words:
            if x == 'PAD':
                continue
            out_string.append(x)
        print('Bot:', ' '.join(out_string))


run()
