import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.linalg import VectorUDT, SparseVector
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
import pyspark
from redisai.model import save_sparkml
from redisai.model import onnx_tensortypes


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
data = original_data.selectExpr(
    "cast(label as string) as label", "truncateFeatures(features) as features")
breakpoint()
label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel", handleInvalid='error')
feature_indexer = VectorIndexer(
    inputCol="features", outputCol="indexedFeatures", maxCategories=10, handleInvalid='error')
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])
model = pipeline.fit(data)
labeltype = ('label', onnx_tensortypes.StringTensorType([1, 1]))
featurestype = ('features', onnx_tensortypes.FloatTensorType([1, feature_count]))
save_sparkml(model, 'spark.onnx', [labeltype, featurestype], spark_session=spark)
