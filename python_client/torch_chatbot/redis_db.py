from os.path import dirname

import numpy as np
import redis

import utils


class DB:
    
    def __init__(self, host='localhost', port=6379, db=0):
        self.max_len = 10
        self.exec = redis.Redis(host=host, port=port, db=db).execute_command

    def initiate(self):
        encoder_path = f'{dirname(dirname(dirname(__file__)))}/models/chatbot/encoder.pt'
        decoder_path = f'{dirname(dirname(dirname(__file__)))}/models/chatbot/decoder.pt'
        with open(encoder_path, 'rb') as f:
            en_model = f.read()
        with open(decoder_path, 'rb') as f:
            de_model = f.read()
        self.exec('AI.MODELSET', 'encoder', 'TORCH', 'CPU', en_model)
        self.exec('AI.MODELSET', 'decoder', 'TORCH', 'CPU', de_model)

    def process(self, nparray, length):
        self.exec(
            'AI.TENSORSET', 'sentence', 'INT64',
            *nparray.shape, 'BLOB', nparray.tobytes())
        self.exec(
            'AI.TENSORSET', 'length', 'INT64',
            *length.shape, 'BLOB', length.tobytes())
        self.exec(
            'AI.MODELRUN', 'encoder', 'INPUTS', 'sentence', 'length',
            'OUTPUTS', 'e_output', 'd_hidden')
        self.exec(
            'AI.TENSORSET', 'd_input', 'INT64', 1, 1, 'VALUES', utils.SOS_token)
        i = 0
        out = []
        while i < self.max_len:
            i += 1
            self.exec(
                'AI.MODELRUN', 'decoder', 'INPUTS', 'd_input', 'd_hidden', 'e_output',
                'OUTPUTS', 'd_output', 'd_hidden')
            d_output = self.exec('AI.TENSORGET', 'd_output', 'BLOB')[2]
            d_output_ret = np.frombuffer(d_output, dtype=np.float32)
            d_output_ret = d_output_ret.reshape(1, utils.voc.num_words)
            ind = int(d_output_ret.argmax())
            if ind == utils.EOS_token:
                break
            self.exec(
            'AI.TENSORSET', 'd_input', 'INT64', 1, 1, 'VALUES', ind)
            if ind == utils.PAD_token:
                continue
            out.append(ind)
        return utils.indices2str(out)


if __name__ == '__main__':
    redis_db = DB()