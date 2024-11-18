import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, center_box=(-1, 1), random_state=42)

num_projections = 5
mean = [0, 0]
cov = [[1, 0], [0, 1]]  
projections = np.random.multivariate_normal(mean, cov, num_projections)
projections /= np.linalg.norm(projections, axis=1, keepdims=True)

num_bins = 20 ## Schimb num_bins din aceasta variabila si vad diferentele

## Histogramele vor avea mai puține intervale (bin-uri), ceea ce înseamnă că valorile proiectate vor fi distribuite în intervale largi.
## Histogramele vor avea mai multe bin-uri, oferind o rezoluție mai detaliată a distribuției datelor proiectate.
histogram_range = (-10, 10)  
scores = np.zeros(n_samples)

for proj in projections:
    projected_values = X @ proj 
    hist, bin_edges = np.histogram(projected_values, bins=num_bins, range=histogram_range, density=True)
    probabilities = np.interp(projected_values, bin_edges[:-1], hist) 
    scores += probabilities

scores /= num_projections

test_X = np.random.uniform(-3, 3, (n_samples, 2))

test_scores = np.zeros(n_samples)
for proj in projections:
    projected_values = test_X @ proj
    hist, bin_edges = np.histogram(projected_values, bins=num_bins, range=histogram_range, density=True)
    probabilities = np.interp(projected_values, bin_edges[:-1], hist)
    test_scores += probabilities

test_scores /= num_projections

plt.scatter(test_X[:, 0], test_X[:, 1], c=test_scores, cmap='viridis', s=10)
plt.colorbar(label='Anomaly Score')
plt.title('Anomaly Scores of Test Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
