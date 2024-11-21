from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.make_moons(n_samples=1000, noise=0.05)

clusters = AgglomerativeClustering(n_clusters=2, linkage='single').fit(X)
predictions = clusters.labels_
colors = np.array(['red', 'blue'])
plt.scatter(X[:, 0], X[:, 1], color=colors[predictions])
plt.title("Agglomerative clustering")
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
predictions = kmeans.labels_
colors = np.array(['red', 'blue'])
plt.scatter(X[:, 0], X[:, 1], color=colors[predictions])
plt.title("K-Means")
plt.show()

# TODO compare agglomerative vs. kmeans predictions on same plot :)
