import redisai as rai
from redisai import load_model

model = load_model("../models/ONNX/boston.onnx")
con = rai.Client()
con.modelset("onnx_model", rai.Backend.onnx, rai.Device.cpu, model)

# dummydata taken from sklearn.datasets.load_boston().data[0]
dummydata = [
    0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]
tensor = rai.Tensor.scalar(rai.DType.float, *dummydata)
con.tensorset("input", tensor)

con.modelrun("onnx_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(f"House cost predicted by model is ${outtensor.to_numpy().item() * 1000}")
