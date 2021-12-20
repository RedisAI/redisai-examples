import string
import ml2rt
import redisai as rai
import numpy as np
from cli import arguments

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
all_characters = string.printable

hidden_size = 300
n_layers = 2
batch_size = 1

filepath = '../models/pytorch/charrnn/charrnn_pipeline.pt'


def int2str(int_data):
    return ''.join([all_characters[i] for i in int_data])


model = ml2rt.load_model(filepath)

out1 = con.modelset('charRnn', 'torch', device, model)
hidden = np.zeros((n_layers, batch_size, hidden_size), dtype=np.float32)
out2 = con.tensorset('hidden', hidden)
prime_tensor = np.array([5], np.int64)
out3 = con.tensorset('prime', prime_tensor)
out4 = con.modelrun('charRnn', ['prime', 'hidden'], ['out'])
out5 = con.tensorget('out')
para = int2str(out5)
print(para)
