import os
import sys
import numpy
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, OneVsRest


executable = sys.executable
os.environ["SPARK_HOME"] = pyspark.__path__[0]
os.environ["PYSPARK_PYTHON"] = executable
os.environ["PYSPARK_DRIVER_PYTHON"] = executable
spark = SparkSession.builder.appName("redisai_trial").getOrCreate()

data = spark.read.format("libsvm").load('multiclass_classification_data.txt')
testdata = spark.createDataFrame(
    [(Vectors.dense([-0.222222, 0.5, -0.762712, -0.833333]),)], ["features"])
lr = LogisticRegression(maxIter=100, tol=0.0001, regParam=0.01)
ovr = OneVsRest(classifier=lr)
model = ovr.fit(data)
predicted = model.transform(testdata)
output = predicted.toPandas().prediction.values.astype(numpy.float32)
print(output)
