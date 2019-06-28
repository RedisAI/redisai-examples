import redisai as rai
from redisai import load_model
model = load_model("../models/spark/linear_regression/linear_regression.onnx")

con = rai.Client()
con.modelset("spark_model", rai.Backend.onnx, rai.Device.cpu, model, input=['features'])
dummydata = [15.0]
tensor = rai.Tensor.scalar(rai.DType.float, *dummydata)
con.tensorset("input", tensor)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(outtensor.to_numpy())
