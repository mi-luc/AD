import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import zscore

np.random.seed(42)
n_samples = 1000
contamination_rate = 0.1

mean_1, mean_2 = 5, 10
var_1, var_2 = 1, 4


X1 = np.random.normal(loc=mean_1, scale=np.sqrt(var_1), size=n_samples)

X2 = np.random.normal(loc=mean_2, scale=np.sqrt(var_2), size=n_samples)

X_train = np.column_stack((X1, X2))

z_scores = zscore(X_train, axis=0)  

z_scores_combined = np.sqrt(z_scores[:, 0]**2 + z_scores[:, 1]**2)

z_threshold = np.quantile(z_scores_combined, 1 - contamination_rate)











y_train_pred = (z_scores_combined >= z_threshold).astype(int) 

y_train = np.zeros(n_samples, dtype=int)
n_anomalies = int(contamination_rate * n_samples)
y_train[-n_anomalies:] = 1

balanced_acc = balanced_accuracy_score(y_train, y_train_pred)

print(f"Z-score threshold for top 10% contamination: {z_threshold:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
