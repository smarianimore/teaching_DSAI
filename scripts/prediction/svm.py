import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("data/sonar.csv", header=None)

    y = df.iloc[:,-1]  # take all the rows and the last column
    X = df.iloc[:,:-1]  # take all the rows and all the columns except the last one

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    hyperparameters = [
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
    ]

    svc = svm.SVC()
    clf = GridSearchCV(svc, hyperparameters)
    clf.fit(X_train, y_train)

    print("Best parameters set found on train set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on train set (mean (std.dev.)):")
    print()
    means = clf.cv_results_['mean_test_score']  # do not be fooled: "mean_test_score" is the nam used by the library, but in this case the score has been computed on training!
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))  # string formatting "magic", see https://docs.python.org/3/library/stdtypes.html#old-string-formatting

    y_pred = clf.predict(X_test)
    print("---------------------------")
    print("CONFUSION MATRIX (test set)")
    print("---------------------------")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------")
    print("Classification report (test set)")
    print("--------------------------------")
    print(classification_report(y_test, y_pred))
