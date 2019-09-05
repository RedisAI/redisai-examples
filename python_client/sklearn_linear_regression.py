import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/sklearn/linear_regression/linear_regression.onnx")

if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("sklearn_model", rai.Backend.onnx, device, model)

# dummydata taken from sklearn.datasets.load_boston().data[0]
dummydata = [
    0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]
tensor = rai.Tensor.scalar(rai.DType.float, *dummydata)
con.tensorset("input", tensor)

con.modelrun("sklearn_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(f"House cost predicted by model is ${outtensor.to_numpy().item() * 1000}")
