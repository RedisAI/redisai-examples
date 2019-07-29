import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.ml.feature import VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
import pyspark
from ml2rt import save_sparkml
from ml2rt import utils


executable = sys.executable
os.environ["SPARK_HOME"] = pyspark.__path__[0]
os.environ["PYSPARK_PYTHON"] = executable
os.environ["PYSPARK_DRIVER_PYTHON"] = executable
spark = SparkSession.builder.appName("redisai_trial").getOrCreate()
original_data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
feature_count = 5
spark.udf.register(
    "truncateFeatures",
    lambda x: SparseVector(feature_count, range(0, feature_count), x.toArray()[125: 130]),
    VectorUDT())
data = original_data.selectExpr("label", "truncateFeatures(features) as features")
feature_indexer = VectorIndexer(
    inputCol="features", outputCol="indexedFeatures", maxCategories=4, handleInvalid='error')
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[feature_indexer, dt])

# (trainingData, testData) = data.randomSplit([0.9, 0.1])
model = pipeline.fit(data)
featurestype = utils.guess_onnx_tensortype(
    node_name='features', dtype='float32', shape=(1, feature_count))
save_sparkml(
    model, 'spark.onnx', initial_types=[featurestype], spark_session=spark)
