import numpy as np
import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/sklearn/logistic_regression/logistic.onnx")

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("sklearn_model", 'onnx', device, model)

dummydata = np.array([[6.9, 3.1, 5.4, 2.1]], dtype=np.float32)
con.tensorset("input", dummydata)

# dummy output because by default sklearn logistic regression outputs
# value and probability. Since RedisAI doesn't support specifying required
# outputs now, we need to keep placeholders for all the default outputs.
con.modelrun("sklearn_model", ["input"], ["output", "dummy"])
outtensor = con.tensorget("output")
print(f" Output class: {outtensor.item()}")
