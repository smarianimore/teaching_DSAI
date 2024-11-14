import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_NN(i_layer):
    model = Sequential()
    model.add(Dense(i_layer, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == "__main__":

    df = pd.read_csv("data/bank_customer_churn.csv", index_col="customer_id")

    one_hot_country = pd.get_dummies(df.country, prefix='country')
    one_hot_gender = pd.get_dummies(df.gender, prefix='gender')
    df = df.drop(["country","gender"],axis=1)
    df = pd.concat([df, one_hot_country, one_hot_gender], axis=1)

    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = create_NN(X_train.shape[1])

    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(X_train, y_train, epochs=100, validation_split=0.2)
    y_pred = model.predict(X_test)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    print("---------------------------")
    print("CONFUSION MATRIX (test set)")
    print("---------------------------")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")
    print("Classification report (test set)")
    print("--------------------------------")
    print(classification_report(y_test, y_pred))

    model_early = create_NN(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=10)
    model_early.compile(loss='binary_crossentropy', optimizer='adam')

    history = model_early.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

    y_pred = model_early.predict(X_test)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    print("---------------------------")
    print("CONFUSION MATRIX (early stopping, test set)")
    print("---------------------------")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")
    print("Classification report (early stopping, test set)")
    print("--------------------------------")
    print(classification_report(y_test, y_pred))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.title("Loss (with early stopping)")
    plt.show()

