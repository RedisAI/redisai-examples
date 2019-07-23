import numpy as np

from redisai import BlobTensor, Client, Backend, Device
from ml2rt import load_model


tensor = BlobTensor.from_numpy(np.ones((1, 13), dtype=np.float32))
model = load_model('../models/sklearn/boston_house_price_prediction/boston.onnx')

con = Client()
con.tensorset('tensor', tensor)
con.modelset('model', Backend.onnx, Device.cpu, model)
con.modelrun('model', input=['tensor'], output=['out'])
out = con.tensorget('out', as_type=BlobTensor)
print(out.to_numpy())
