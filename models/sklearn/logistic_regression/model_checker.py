import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Input shape:', X_test.shape)

logistic = LogisticRegression(solver='liblinear', multi_class='auto')
model = logistic.fit(X_train, y_train)
test_data = np.array([[5., 3.6, 1.4, 0.2]])
prediction = model.predict(test_data)
print(prediction)
