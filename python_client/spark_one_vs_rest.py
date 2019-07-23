import numpy as np
import redisai as rai
from ml2rt import load_model

model = load_model("../models/spark/one_vs_rest/spark.onnx")

con = rai.Client()
con.modelset("spark_model", rai.Backend.onnx, rai.Device.cpu, model)
dummydata = np.array(
    [[-0.222222, 0.5, -0.762712, -0.833333]], dtype=np.float32)
tensor = rai.BlobTensor.from_numpy(dummydata)
con.tensorset("input", tensor)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(outtensor.to_numpy())
