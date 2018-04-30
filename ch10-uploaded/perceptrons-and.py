import numpy as np
from sklearn.linear_model import Perceptron

X = [[0, 0.1], [0.1, 0.9], [0.02, 0], [0.9, 0], [1, 0.9]]
y = [0, 0, 0, 0, 1]

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

#y_pred = per_clf.predict([[2, 0.5]])
y_pred = per_clf.predict([[0.12, 0], [1, 0], [0, 1], [0.9, 0.9], [1, 1]])

print(y_pred) # output expected to be [0 0 0 1 1]
