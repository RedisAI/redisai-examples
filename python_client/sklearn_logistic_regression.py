import numpy as np
import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/sklearn/logistic_regression/logistic.onnx")

if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("sklearn_model", rai.Backend.onnx, device, model)

dummydata = np.array([[6.9, 3.1, 5.4, 2.1]], dtype=np.float32)
tensor = rai.BlobTensor.from_numpy(dummydata)
con.tensorset("input", tensor)

# dummy output because by default sklearn logistic regression outputs
# value and probability. Since RedisAI doesn't support specifying required
# outputs now, we need to keep placeholders for all the default outputs.
con.modelrun("sklearn_model", ["input"], ["output", "dummy"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(f" Output class: {outtensor.to_numpy().item()}")
