from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from redisai import save_sklearn
from redisai import onnx_utils
from redisai import DType


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Input shape:', X_test.shape)

logistic = LogisticRegression(solver='liblinear', multi_class='auto')
logistic.fit(X_train, y_train)


initial_type = onnx_utils.get_tensortype(
    node_name='float_input', dtype=DType.float32, shape=(1, 4))

save_sklearn(logistic, 'logistic.onnx', initial_types=[initial_type])
