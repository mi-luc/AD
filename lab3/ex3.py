import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

data = scipy.io.loadmat("shuttle.mat")
X = data['X']
y = data['y'].ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

iforest_ba_scores, iforest_roc_auc_scores = [], []
dif_ba_scores, dif_roc_auc_scores = [], []
loda_ba_scores, loda_roc_auc_scores = [], []

for _ in range(10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None)
    
    iforest = IForest(contamination=0.1)
    iforest.fit(X_train)
    iforest_preds = iforest.predict(X_test)
    iforest_scores = iforest.decision_function(X_test)
    
    iforest_ba = balanced_accuracy_score(y_test, iforest_preds)
    iforest_roc_auc = roc_auc_score(y_test, iforest_scores)
    iforest_ba_scores.append(iforest_ba)
    iforest_roc_auc_scores.append(iforest_roc_auc)
    
    dif = DIF(contamination=0.1, hidden_neurons=(16, 8))
    dif.fit(X_train)
    dif_preds = dif.predict(X_test)
    dif_scores = dif.decision_function(X_test)
    
    dif_ba = balanced_accuracy_score(y_test, dif_preds)
    dif_roc_auc = roc_auc_score(y_test, dif_scores)
    dif_ba_scores.append(dif_ba)
    dif_roc_auc_scores.append(dif_roc_auc)
    
    loda = LODA(contamination=0.1)
    loda.fit(X_train)
    loda_preds = loda.predict(X_test)
    loda_scores = loda.decision_function(X_test)
    
    loda_ba = balanced_accuracy_score(y_test, loda_preds)
    loda_roc_auc = roc_auc_score(y_test, loda_scores)
    loda_ba_scores.append(loda_ba)
    loda_roc_auc_scores.append(loda_roc_auc)

mean_iforest_ba = np.mean(iforest_ba_scores)
mean_iforest_roc_auc = np.mean(iforest_roc_auc_scores)
mean_dif_ba = np.mean(dif_ba_scores)
mean_dif_roc_auc = np.mean(dif_roc_auc_scores)
mean_loda_ba = np.mean(loda_ba_scores)
mean_loda_roc_auc = np.mean(loda_roc_auc_scores)

print(f"Isolation Forest - Mean BA: {mean_iforest_ba:.4f}, Mean ROC AUC: {mean_iforest_roc_auc:.4f}")
print(f"Deep Isolation Forest - Mean BA: {mean_dif_ba:.4f}, Mean ROC AUC: {mean_dif_roc_auc:.4f}")
print(f"LODA - Mean BA: {mean_loda_ba:.4f}, Mean ROC AUC: {mean_loda_roc_auc:.4f}")
