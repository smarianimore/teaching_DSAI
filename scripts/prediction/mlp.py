import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

    clf = MLPClassifier(early_stopping=True, solver='adam')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("---------------------------")
    print("CONFUSION MATRIX (test set)")
    print("---------------------------")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")
    print("Classification report (test set)")
    print("--------------------------------")
    print(classification_report(y_test, y_pred))
