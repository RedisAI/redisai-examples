import redis
import numpy as np
import string

all_characters = string.printable

r = redis.Redis(host='localhost', port=6379, db=0)
hidden_size = 300
n_layers = 2
batch_size = 1

filepath = '../models/pytorch/charrnn/charrnn_pipeline.pt'


def int2str(int_data):
    return ''.join([all_characters[i] for i in int_data])


with open(filepath, 'rb') as f:
    model = f.read()

out1 = r.execute_command('AI.MODELSET', 'charRnn', 'TORCH', 'CPU', model)
hidden = np.zeros((n_layers, batch_size, hidden_size), dtype=np.float32)
out2 = r.execute_command(
    'AI.TENSORSET', 'hidden', 'FLOAT',
    n_layers, batch_size, hidden_size,
    'BLOB', hidden.tobytes())
out3 = r.execute_command('AI.TENSORSET', 'prime', 'INT64', 1, 'VALUES', 5)
out4 = r.execute_command('AI.MODELRUN', 'charRnn', 'INPUTS', 'prime', 'hidden', 'OUTPUTS', 'out')
out5 = r.execute_command('AI.TENSORGET', 'out', 'VALUES')
para = int2str(out5[2])
print(para)
