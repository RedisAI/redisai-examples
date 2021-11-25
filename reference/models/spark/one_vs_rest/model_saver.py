import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, OneVsRest
from ml2rt import save_sparkml
from ml2rt import utils


executable = sys.executable
os.environ["SPARK_HOME"] = pyspark.__path__[0]
os.environ["PYSPARK_PYTHON"] = executable
os.environ["PYSPARK_DRIVER_PYTHON"] = executable
spark = SparkSession.builder.appName("redisai_trial").getOrCreate()

data = spark.read.format("libsvm").load('multiclass_classification_data.txt')
lr = LogisticRegression(maxIter=100, tol=0.0001, regParam=0.01)
ovr = OneVsRest(classifier=lr)
model = ovr.fit(data)
feature_count = data.first()[1].size
tensor_types = utils.guess_onnx_tensortype(
    node_name='features', dtype='float32', shape=(1, feature_count))
save_sparkml(model, 'spark.onnx', initial_types=[tensor_types])
