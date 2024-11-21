import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure

if __name__ == "__main__":

    data = datasets.load_iris()
    print(f"{data['feature_names']=}")
    X, y = datasets.load_iris(return_X_y=True)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print(f"{kmeans.cluster_centers_=}")
    print(f"{kmeans.labels_=}")
    print(f"{homogeneity_completeness_v_measure(y, kmeans.labels_)=}")

    predictions = kmeans.labels_
    colors = np.array(['red', 'blue', 'green'])
    plt.scatter(X[:, 2], X[:, 3], color=colors[predictions])  # we only plot the last two features (2D plot :/)
    plt.title("Predictions")
    plt.show()

    predictions = kmeans.labels_
    colors = np.array(['red', 'blue', 'green'])
    markers = ['o', 'x', 's']
    for i in range(len(X)):  # inefficient, but simple (we plot one datapoint at a time)
        plt.scatter(X[i,2], X[i,3], marker=markers[y[i]], color=colors[predictions[i]])  # NOTICE: markers are based on ground truth, colors on prediction -> easy to see errors!
    plt.title("Predictions (colour) vs ground truth (marker)")
    plt.show()
