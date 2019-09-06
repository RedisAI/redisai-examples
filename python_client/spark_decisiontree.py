import numpy as np
import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/spark/decisiontree_with_pipeline/spark.onnx")

if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("spark_model", rai.Backend.onnx, device, model)
features = np.array([[0., 0., 0., 0., 0.], [0., 0., 0., 17., 206.]], dtype=np.float32)
tensor = rai.BlobTensor.from_numpy(features)
con.tensorset("input", tensor)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output", as_type=rai.BlobTensor)
print(outtensor.to_numpy())
