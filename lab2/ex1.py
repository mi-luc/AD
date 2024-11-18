import numpy as np
import matplotlib.pyplot as plt

a, b, c = 1.5, -2.0, 0.5
num_points = 100
variances_x=[[-3,3],[-20,20],[-3,3],[-20,20]]
variances = [1, 1, 20,20] 
i=0
titles = [
    "LOW X, LOW epsilon",
    "HIGH X, LOW epsilon",
    "LOW X, HIGH epsilon",
    "HIGH X, HIGH epsilon"
]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs=axs.flat
for variance in variances:
    
    
    x1 = np.random.uniform(variances_x[i][0], variances_x[i][1], num_points)
   
    epsilon = np.random.normal(0, np.sqrt(variance), num_points)
    y = a * x1 + b + epsilon
    
    X = np.vstack([x1, np.ones(num_points)]).T
    X_x_Xt=np.matmul(X.T,X)
    X_x_Xt_inv=np.linalg.inv(X_x_Xt)
    H=np.matmul(X,X_x_Xt_inv)
    H=np.matmul(H,X.T)
   
    leverage_scores = np.diag(H)
    
    high_leverage_points = leverage_scores > np.percentile(leverage_scores, 80)
    
    axs[i].scatter(x1, y, label="Data Points", color="blue", alpha=0.6)
    axs[i].scatter(x1[high_leverage_points], y[high_leverage_points], color="green", label="High Leverage Points")
    
    # Set titles and labels
    axs[i].set_title(titles[i])
    axs[i].set_xlabel("x1")
    axs[i].set_ylabel("y")
    axs[i].legend()
    i+=1

x1_2d = np.random.uniform(-10, 10, num_points)
x2 = np.random.uniform(-10, 10, num_points)
epsilon_2d = np.random.normal(0, np.sqrt(variance), num_points)
y_2d = a * x1_2d + b * x2 + c + epsilon_2d

X_2d = np.vstack([x1_2d, x2, np.ones(num_points)]).T
H_2d = X_2d @ np.linalg.inv(X_2d.T @ X_2d) @ X_2d.T
leverage_scores_2d = np.diag(H_2d)
high_leverage_points_2d = leverage_scores_2d > np.percentile(leverage_scores_2d, 90)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1_2d, x2, y_2d, label="Data Points")
ax.scatter(x1_2d[high_leverage_points_2d], x2[high_leverage_points_2d], y_2d[high_leverage_points_2d],
           color="green")
ax.set_title(f"2D Linear Model")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.legend()
plt.show()
