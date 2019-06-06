from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from redisai.model import Model as raimodel

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = sklearn.metrics.mean_squared_error(y_test, pred)
print("Mean Squared Error: ", mse)

# 1 is batch size and 13 is num features
#   reference: https://github.com/onnx/sklearn-onnx/blob/master/skl2onnx/convert.py
initial_type = [('float_input', FloatTensorType([1, 13]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)
raimodel.save(onnx_model, 'boston.onnx')
