import numpy as np
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.data import generate_data
from scipy.stats import zscore

X_train, _, y_train, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1, random_state=52)

z_scores = zscore(X_train)

#print(z_scores)


contamination_rate = 0.1
z_threshold = np.quantile(z_scores, 1 - contamination_rate)


print(f"Z-score threshold: {z_threshold}")

y_train_pred=[]

for point in z_scores:
    if abs(point) >= z_threshold:
        y_train_pred.append(1)
    else:
        y_train_pred.append(0)

#print(y_train_pred)

balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

print(f"Balanced Accuracy: {balanced_acc:.4f}")
