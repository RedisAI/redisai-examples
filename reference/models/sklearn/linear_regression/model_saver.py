from sklearn.linear_model import LinearRegression
import numpy as np

from ml2rt import save_sklearn

# y = x + 1
x = np.array([[1.0], [2.0], [6.0], [4.0], [3.0], [5.0]])
y = np.array([[2.0], [3.0], [7.0], [5.0], [4.0], [6.0]])


model = LinearRegression()
model.fit(x, y)

# Converting to float32 since onnx-sklearn doesnt support double yet
dummy = np.array([[1.0]], dtype=np.float32)
save_sklearn(model, 'linear_regression.onnx', prototype=dummy)
