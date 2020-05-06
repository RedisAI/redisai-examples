import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/spark/linear_regression/linear_regression.onnx")

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("spark_model", 'onnx', device, model, inputs=['features'])
dummydata = [15.0]
con.tensorset("input", dummydata, shape=(1, 1), dtype='float32')
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output")
print(outtensor)
