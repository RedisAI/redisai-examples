import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from ml2rt import save_sparkml
from ml2rt import utils


executable = sys.executable
os.environ["SPARK_HOME"] = pyspark.__path__[0]
os.environ["PYSPARK_PYTHON"] = executable
os.environ["PYSPARK_DRIVER_PYTHON"] = executable
spark = SparkSession.builder.appName("redisai_trial").getOrCreate()


data = spark.createDataFrame([
    (Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
    (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
    (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)
], ["features"])
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(data)

feature_count = data.first()[0].size
N = data.count()

featurestype = utils.guess_onnx_tensortype(
    node_name='features', dtype='float32', shape=(N, feature_count))
save_sparkml(model, 'spark.onnx', initial_types=[featurestype], spark_session=spark)
