import numpy as np
from sklearn.linear_model import Perceptron

X = [[0], [0.1], [0.2], [0.9], [1]]
y = [1, 1, 1, 0, 0]

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

#y_pred = per_clf.predict([[2, 0.5]])
y_pred = per_clf.predict([[0.12], [0.9]])

print(y_pred)
