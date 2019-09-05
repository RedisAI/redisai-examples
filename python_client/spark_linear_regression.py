import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/spark/linear_regression/linear_regression.onnx")

if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("spark_model", rai.Backend.onnx, device, model, input=['features'])
dummydata = [15.0]
tensor = rai.Tensor.scalar(rai.DType.float, *dummydata)
con.tensorset("input", tensor)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(outtensor.to_numpy())
