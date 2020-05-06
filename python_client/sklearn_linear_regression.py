import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/sklearn/linear_regression/linear_regression.onnx")

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("sklearn_model", 'onnx', device, model)

# dummydata taken from sklearn.datasets.load_boston().data[0]
dummydata = [15.0]
con.tensorset("input", dummydata, dtype='float32', shape=(1, 1))

con.modelrun("sklearn_model", ["input"], ["output"])
outtensor = con.tensorget("output")
print(f"House cost predicted by model is ${outtensor.item() * 1000}")
