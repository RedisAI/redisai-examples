import numpy as np

import redisai
from ml2rt import load_model

from cli import arguments


tensor = np.ones((1, 13), dtype=np.float32)
model = load_model('../models/sklearn/boston_house_price_prediction/boston.onnx')

if arguments.gpu:
    device = 'GPU'
else:
    device = 'CPU'

con = redisai.Client(host=arguments.host, port=arguments.port)
con.tensorset('tensor', tensor)
con.modelset('model', 'onnx', device, model)
con.modelrun('model', inputs=['tensor'], outputs=['out'])
out = con.tensorget('out')
print(out)
