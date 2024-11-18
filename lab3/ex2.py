import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

n_samples = 500
centers = [(10, 0), (0, 10)]
X, _ = make_blobs(n_samples=2 * n_samples, centers=centers, cluster_std=1.0, random_state=42)


contamination = 0.02
iforest = IForest(contamination=contamination, random_state=42)
iforest.fit(X)


test_X = np.random.uniform(-10, 20, (1000, 2))


iforest_scores = iforest.decision_function(test_X)


dif = DIF(contamination=contamination, hidden_neurons=(16, 8))
dif.fit(X)
dif_scores = dif.decision_function(test_X)

loda = LODA(contamination=contamination)
loda.fit(X)
loda_scores = loda.decision_function(test_X)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].scatter(test_X[:, 0], test_X[:, 1], c=iforest_scores, cmap='viridis', s=10)
axes[0].set_title('Isolation Forest')
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')

axes[1].scatter(test_X[:, 0], test_X[:, 1], c=dif_scores, cmap='viridis', s=10)
axes[1].set_title('Deep Isolation Forest (DIF)')
axes[1].set_xlabel('X-axis')
axes[1].set_ylabel('Y-axis')

axes[2].scatter(test_X[:, 0], test_X[:, 1], c=loda_scores, cmap='viridis', s=10)
axes[2].set_title('LODA')
axes[2].set_xlabel('X-axis')
axes[2].set_ylabel('Y-axis')

plt.colorbar(axes[2].collections[0], ax=axes, location='right')
plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D


centers_3d = [(0, 10, 0), (10, 0, 10)]
X_3d, _ = make_blobs(n_samples=2 * n_samples, centers=centers_3d, cluster_std=1.0, random_state=42)


test_X_3d = np.random.uniform(-10, 20, (1000, 3))


iforest.fit(X_3d)
iforest_scores_3d = iforest.decision_function(test_X_3d)

dif.fit(X_3d)
dif_scores_3d = dif.decision_function(test_X_3d)

loda.fit(X_3d)
loda_scores_3d = loda.decision_function(test_X_3d)

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

ax1.scatter(test_X_3d[:, 0], test_X_3d[:, 1], test_X_3d[:, 2], c=iforest_scores_3d, cmap='viridis', s=10)
ax1.set_title('Isolation Forest 3D')

ax2.scatter(test_X_3d[:, 0], test_X_3d[:, 1], test_X_3d[:, 2], c=dif_scores_3d, cmap='viridis', s=10)
ax2.set_title('Deep Isolation Forest 3D')

ax3.scatter(test_X_3d[:, 0], test_X_3d[:, 1], test_X_3d[:, 2], c=loda_scores_3d, cmap='viridis', s=10)
ax3.set_title('LODA 3D')

plt.tight_layout()
plt.show()
