from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":

    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)  # HERE LEARNING HAPPENS
    y_pred = knn.predict(X_val)  #HERE WE USE LEARNT MODEL (that does not learn anymore)
    print("---------------------------------")
    print("CONFUSION MATRIX (validation set)")
    print("---------------------------------")
    print(confusion_matrix(y_val, y_pred))
    print("--------------------------------------")
    print("CLASSIFICATION REPORT (validation set)")
    print("--------------------------------------")
    print(classification_report(y_val, y_pred))

    print("-----------------------------")
    print("MANUAL HYPER-PARAMETER TUNING")
    print("-----------------------------")
    best_f1 = 0
    best_k = 0
    for k in [3,5,7,9,11]:  # hyper-parameter tuning
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"{k=} ---> {f1=} (validation set)")
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_trainval, y_trainval)
    y_pred = knn.predict(X_test)
    print(f"{f1_score(y_test, y_pred)=} (test set)")

    print("--------------------------------")
    print("AUTOMATIC HYPER-PARAMETER TUNING")
    print("--------------------------------")
    from sklearn.model_selection import GridSearchCV  # AUTOMATE HYPER-PARAMETER TUNING :)

    knn = KNeighborsClassifier()
    k_range = [1,3,5,7,9,11,13,15]
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='f1', verbose=1)  # cv=10 means 10-fold cross-validation, scoring indicates what metric to use to establish best model
    grid_search = grid.fit(X_trainval, y_trainval)

    print(f"{grid_search.best_params_=} (train+validation set)")
    knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_["n_neighbors"])
    knn.fit(X_trainval, y_trainval)

    y_pred = knn.predict(X_test)
    print(f"{f1_score(y_test, y_pred)=} (test set)")
