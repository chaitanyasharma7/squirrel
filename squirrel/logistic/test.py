from logistic import LogisticRegression
import numpy as np

y = np.array([1, 0, 1, 1, 0, 1, 0, 0]).reshape((1, -1))
a = np.array([0.9, 0.1, 0.1, 0.3, 0.1, 0.8, 0.3, 0.2]).reshape((1, -1))

#print(np.squeeze(LogisticRegression.log_loss(a=a, y=y)))

print(np.sum(np.square((a))))
