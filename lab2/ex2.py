from pyod.utils.data import generate_data_clusters as gen_data
from pyod.models.knn import KNN
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = gen_data(n_train=400, n_test=200, n_clusters=2, contamination=0.15,random_state=42)

KNN_model = KNN(contamination=0.15, n_neighbors=12)
KNN_model.fit(X_train)


y_pred_train = KNN_model.predict(X_train) 
y_pred_test = KNN_model.predict(X_test)    


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flat

for i in range(4):
   
    if i==0:
        axs[i].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label="Inliers", color="blue", alpha=0.6)
        axs[i].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label="Inliers", color="red", alpha=0.6)
        axs[i].set_title("Ground truth labels for training data")
    
    if i==1:
        axs[i].scatter(X_train[y_pred_train == 0, 0], X_train[y_pred_train == 0, 1], label="Inliers", color="blue", alpha=0.6)
        axs[i].scatter(X_train[y_pred_train == 1, 0], X_train[y_pred_train == 1, 1], label="Inliers", color="red", alpha=0.6)
        axs[i].set_title("Predicted labels for training data")
   
    if i==2:
        axs[i].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label="Inliers", color="blue", alpha=0.6)
        axs[i].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label="Inliers", color="red", alpha=0.6)
        axs[i].set_title("Ground truth labels for test data")
  
    if i==3:
        axs[i].scatter(X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], label="Inliers", color="blue", alpha=0.6)
        axs[i].scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], label="Inliers", color="red", alpha=0.6)
        axs[i].set_title("Predicted labels for test data")
    
   
    axs[i].set_xlabel("x")
    axs[i].set_ylabel("y")
    axs[i].legend()

plt.tight_layout()
plt.show()
