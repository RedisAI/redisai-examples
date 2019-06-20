from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn

from redisai import save_model

boston = load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = sklearn.metrics.mean_squared_error(y_test, pred)
print("Mean Squared Error: ", mse)

dummy = X_train[0].reshape(1, -1)
save_model(model, 'boston.onnx', prototype=dummy)
