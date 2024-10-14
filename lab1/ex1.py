import matplotlib.pyplot
import pyod
from pyod.utils.data import generate_data
import matplotlib

import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=53)


plt.figure(figsize=(8, 6))
print(X_train.shape)
print(y_train.shape)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='b', label='Normal')

# Plot outliers
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='r', label='Outliers')

# Add labels and title
plt.title('Scatter plot of training data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()