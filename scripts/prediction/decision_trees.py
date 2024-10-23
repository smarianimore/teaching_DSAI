import pydotplus
from matplotlib import pyplot as plt
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

X, y = datasets.load_wine(return_X_y=True)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.3)

dt = DecisionTreeClassifier()
dt.fit(X_trainval, y_trainval)
y_pred = dt.predict(X_test)
print("--------------------------------")
print("CLASSIFICATION REPORT (test set)")
print("--------------------------------")
print(classification_report(y_test, y_pred))

print("---------------------")
print("LEARNT TREE (textual)")
print("---------------------")
print(tree.export_text(dt, feature_names=datasets.load_wine().feature_names))

print("-----------------------")
print("LEARNT TREE (graphical)")
print("-----------------------")
fig = plt.figure()
tree.plot_tree(dt, filled=True, feature_names=datasets.load_wine().feature_names)
plt.show()
#fig.savefig("decision_tree.pdf")

print("------------------------------")
print("LEARNT TREE (GraphViz package)")
print("------------------------------")
dot_data = tree.export_graphviz(dt, filled=True, feature_names=datasets.load_wine().feature_names)
pydot_graph = pydotplus.graph_from_dot_data(dot_data)
pydot_graph.write_pdf('wine_dt.pdf')

print("-------------")
print("RANDOM FOREST")
print("-------------")

from sklearn.ensemble import RandomForestClassifier

dt = RandomForestClassifier()
dt.fit(X_trainval, y_trainval)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))

print("-----------------")
print("GRADIENT BOOSTING")
print("-----------------")

from sklearn.ensemble import GradientBoostingClassifier

dt = GradientBoostingClassifier()
dt.fit(X_trainval, y_trainval)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))
