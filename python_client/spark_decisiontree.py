import redisai as rai
from redisai.model import load_model
model = load_model("../models/spark/decisiontree_with_pipeline/dtree.onnx")

con = rai.Client()
con.modelset("spark_model", rai.Backend.onnx, rai.Device.cpu, model)
label = [1.0]
features = [1.0, 1.0, 1.0, 1.0, 1.0]
ltensor = rai.Tensor.scalar(rai.DType.float, *label)
ftensor = rai.Tensor.scalar(rai.DType.float, *features)
con.tensorset("linput", ltensor)
con.tensorset("finput", ftensor)
con.modelrun("spark_model", ["linput", "finput"], ["output1", "out2", "out3"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(outtensor.to_numpy())
