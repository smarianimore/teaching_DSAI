import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":

    df = pd.read_csv("data/tesla-stock-price.csv", sep=",")

    data = df["close"].values.astype('float32')
    timesteps = data.astype("datetime64[ns]")

    X = []
    y = []
    t = []

    W = 10  # window size (number of days to look back)

    for i in range(W, data.shape[0]):  # the chosen window "slides" over the data 1 day at a time
        X.append(data[i-W:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150, random_state=0, shuffle=False)  # DO NOT SHUFFLE DATA!

    print("*** Random Forest ***")
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    print(mean_absolute_error(y_test, y_pred_rf))

    print("*** Linear Regression ***")
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred_lin = clf.predict(X_test)
    print(mean_absolute_error(y_test, y_pred_lin))

    print("*** KNN Regression ***")
    clf = KNeighborsRegressor(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)
    print(mean_absolute_error(y_test, y_pred_knn))

    plt.plot(y_test, label="actual")
    plt.plot(y_pred_rf, label="random forest")
    plt.plot(y_pred_lin, label="linear")
    plt.plot(y_pred_knn, label="knn (=3)")
    plt.legend()
    plt.show()
