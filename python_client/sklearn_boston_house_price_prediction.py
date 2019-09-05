import sys
import numpy as np

from redisai import BlobTensor, Client, Backend, Device
from ml2rt import load_model

from cli import arguments


tensor = BlobTensor.from_numpy(np.ones((1, 13), dtype=np.float32))
model = load_model('../models/sklearn/boston_house_price_prediction/boston.onnx')

if arguments.gpu:
    device = Device.gpu
else:
    device = Device.cpu

con = Client(host=arguments.host, port=arguments.port)
con.tensorset('tensor', tensor)
con.modelset('model', Backend.onnx, device, model)
con.modelrun('model', input=['tensor'], output=['out'])
out = con.tensorget('out', as_type=BlobTensor)
print(out.to_numpy())
