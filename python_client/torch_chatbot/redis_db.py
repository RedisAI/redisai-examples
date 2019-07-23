from os.path import dirname

import numpy as np
import redis
import redisai as rai
import ml2rt
from . import utils


class DB:

    def __init__(self, host='localhost', port=6379, db=0):
        self.max_len = 10
        self.exec = redis.Redis(host=host, port=port, db=db).execute_command
        self.con = rai.Client(host=host, port=port, db=db)

    def initiate(self):
        encoder_path = f'{dirname(dirname(dirname(__file__)))}/models/pytorch/chatbot/encoder.pt'
        decoder_path = f'{dirname(dirname(dirname(__file__)))}/models/pytorch/chatbot/decoder.pt'
        en_model = ml2rt.load_model(encoder_path)
        de_model = ml2rt.load_model(decoder_path)
        self.con.modelset('encoder', rai.Backend.torch, rai.Device.cpu, en_model)
        self.con.modelset('decoder', rai.Backend.torch, rai.Device.cpu, de_model)

    def process(self, nparray, length):
        tensor = rai.BlobTensor.from_numpy(nparray)
        self.con.tensorset('sentence', tensor)
        length_tensor = rai.BlobTensor.from_numpy(length)
        self.con.tensorset('length', length_tensor)
        self.con.modelrun('encoder', input=['sentence', 'length'], output=['e_output', 'd_hidden'])
        sos_tensor = rai.BlobTensor.from_numpy(
            np.array(utils.SOS_token, dtype=np.int64).reshape(1, 1))
        self.con.tensorset('d_input', sos_tensor)
        i = 0
        out = []
        while i < self.max_len:
            i += 1
            self.con.modelrun(
                'decoder',
                input=['d_input', 'd_hidden', 'e_output'],
                output=['d_output', 'd_hidden'])
            d_output = self.con.tensorget('d_output', as_type=rai.BlobTensor).to_numpy()
            d_output_ret = d_output.reshape(1, utils.voc.num_words)
            ind = int(d_output_ret.argmax())
            if ind == utils.EOS_token:
                break
            inter_tensor = rai.Tensor(rai.DType.int64, shape=[1, 1], value=ind)
            self.con.tensorset('d_input', inter_tensor)
            if ind == utils.PAD_token:
                continue
            out.append(ind)
        return utils.indices2str(out)


if __name__ == '__main__':
    redis_db = DB()
