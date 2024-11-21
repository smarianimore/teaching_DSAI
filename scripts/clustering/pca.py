from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    data = datasets.load_iris()
    print(f"{data['feature_names']=}")
    X, y = datasets.load_iris(return_X_y=True)

    pca = PCA(n_components=2)
    T = pca.fit_transform(X)
    colors = np.array(['red', 'blue', 'green'])
    markers = ['o', 'x', 's']
    for i in range(len(X)):  # inefficient, but simple (we plot one datapoint at a time)
        plt.scatter(T[i,0], T[i,1], color=colors[y[i]])
        plt.scatter(X[i, 2], X[i, 3], marker=markers[y[i]],color="black")
    plt.show()
