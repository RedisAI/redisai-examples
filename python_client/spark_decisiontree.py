import numpy as np
import redisai as rai
from redisai import load_model

model = load_model("../models/spark/decisiontree_with_pipeline/spark.onnx")

con = rai.Client()
con.modelset("spark_model", rai.Backend.onnx, rai.Device.cpu, model)
features = np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 17., 206.]], dtype=np.float32)
tensor = rai.BlobTensor.from_numpy(features)
con.tensorset("input", tensor)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(outtensor.to_numpy())
