import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
import pyspark

executable = sys.executable
os.environ["SPARK_HOME"] = pyspark.__path__[0]
os.environ["PYSPARK_PYTHON"] = executable
os.environ["PYSPARK_DRIVER_PYTHON"] = executable

spark = SparkSession.builder.appName("redisai_trial").getOrCreate()

# label is input + 1
data = spark.createDataFrame([
    (2.0, Vectors.dense(1.0)),
    (3.0, Vectors.dense(2.0)),
    (4.0, Vectors.dense(3.0)),
    (5.0, Vectors.dense(4.0)),
    (6.0, Vectors.dense(5.0)),
    (7.0, Vectors.dense(6.0))
], ["label", "features"])
lr = LinearRegression(maxIter=5, regParam=0.0, solver="normal")
model = lr.fit(data)

# trial
dummy = spark.createDataFrame([(Vectors.dense(1.0), )], ["features"])
prediction = model.transform(dummy)
print(prediction.select("prediction").show())
