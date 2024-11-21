import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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

    model = Sequential()
    model.add(Dense(50, input_shape=(X_train.shape[1],)))
    model.add(Dense(20))
    model.add(Dense(1, activation='linear'))

    es = EarlyStopping(monitor='val_loss', patience=10)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es])

    y_pred = model.predict(X_test)

    plt.plot(y_test, label="actual")
    plt.plot(y_pred, label="keras NN")
    plt.legend()
    plt.show()
