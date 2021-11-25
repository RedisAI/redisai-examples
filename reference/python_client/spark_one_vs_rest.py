import numpy as np
import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/spark/one_vs_rest/spark.onnx")

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("spark_model", 'onnx', device, model)
dummydata = np.array(
    [[-0.222222, 0.5, -0.762712, -0.833333]], dtype=np.float32)
con.tensorset("input", dummydata)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output")
print(outtensor)
