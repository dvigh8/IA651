import numpy as np
from sklearn.linear_model import Perceptron
max_iter=1000
tol=2
X = [[8.5, 9.1], [7.6, 5.2], [0.2, 6.8], [6.6, 6.6], [4.2, 2.4],[3.0,3.1],[9.9,9.9],[2.0,2.2]]
y = [0, 1, 0, 1, 1,0,1,0]

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

#y_pred = per_clf.predict([[2, 0.5]])
y_pred = per_clf.predict([[6, 4], [1, 1], [1,6],[4,6]])

print(y_pred)
