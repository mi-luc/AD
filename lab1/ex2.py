import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42)

contamination_rate = 0.1
model = KNN(contamination=contamination_rate)
model.fit(X_train)

y_train_pred = model.labels_  
y_test_pred = model.predict(X_test)  

y_train_scores = model.decision_scores_ 
y_test_scores = model.decision_function(X_test) 


cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)


tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
tn_test, fp_test, fn_test, tp_test = cm_test.ravel()


balanced_acc_train = balanced_accuracy_score(y_train, y_train_pred)
balanced_acc_test = balanced_accuracy_score(y_test, y_test_pred)

print("Training Confusion Matrix: ", cm_train)
print("Testing Confusion Matrix: ", cm_test)
print(f"Balanced Accuracy (Train): {balanced_acc_train:.4f}")
print(f"Balanced Accuracy (Test): {balanced_acc_test:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
print(fpr.shape)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

contamination_rate_new = 0.2
model_new = KNN(contamination=contamination_rate_new)
model_new.fit(X_train)

y_train_pred_new = model_new.labels_
y_test_pred_new = model_new.predict(X_test)

cm_train_new = confusion_matrix(y_train, y_train_pred_new)
cm_test_new = confusion_matrix(y_test, y_test_pred_new)

balanced_acc_train_new = balanced_accuracy_score(y_train, y_train_pred_new)
balanced_acc_test_new = balanced_accuracy_score(y_test, y_test_pred_new)

print("New Contamination - Balanced Accuracy (Train):", balanced_acc_train_new)
print("New Contamination - Balanced Accuracy (Test):", balanced_acc_test_new)
