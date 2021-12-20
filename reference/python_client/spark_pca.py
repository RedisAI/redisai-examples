import numpy as np
import redisai as rai
from ml2rt import load_model
from cli import arguments

model = load_model("../models/spark/pca/spark.onnx")

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("spark_model", 'onnx', device, model)
dummydata = np.array(
    [[5, 1.0, 7.0, 7.0, 7.0], [2.0, 0.0, 3.0, 4.0, 5.0], [4.0, 0.0, 0.0, 6.0, 7.0]],
    dtype=np.float32)
con.tensorset("input", dummydata)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output")
print(outtensor)
