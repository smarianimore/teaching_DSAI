import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/bank_customer_churn.csv", index_col="customer_id")

one_hot_country = pd.get_dummies(df.country, prefix='country')
one_hot_gender = pd.get_dummies(df.gender, prefix='gender')
df = df.drop(["country","gender"],axis=1)
df = pd.concat([df, one_hot_country, one_hot_gender], axis=1)

y = df["churn"]
X = df.drop("churn", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# normalize features in [0,1] interval (highly suggested for logistic regression)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(class_weight='balanced')  # class_weight='balanced' is used to handle imbalanced classes, where some labels are more frequent than others
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

print("--------------------")
print("Feature coefficients")
print("--------------------")
# quick way to see beta coeficients (weights) for each feature
feature_weights = {f: w for f,w in zip(X.columns, clf.coef_[0])}
print(json.dumps(feature_weights, indent=2))

# investigate impact of feature X (eg age) on Y (churn)
plt.hist(df[df["churn"] == 0]["age"], density=True, histtype='step', bins=20, label="no churn")
plt.hist(df[df["churn"] == 1]["age"], density=True, histtype='step', bins=20, label="churn")
plt.xlabel("age")
plt.ylabel("churn probability")
plt.legend()
plt.show()
