import numpy as np
import redisai as rai
from ml2rt import load_model

from cli import arguments

model = load_model("../models/spark/decisiontree_with_pipeline/spark.onnx")

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)
con.modelset("spark_model", 'onnx', device, model)
features = np.array([[0., 0., 0., 17., 206.]], dtype=np.float32)
con.tensorset("input", features)
con.modelrun("spark_model", ["input"], ["output"])
outtensor = con.tensorget("output")
print(outtensor)
