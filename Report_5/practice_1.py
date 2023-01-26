from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


#clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=[10]) # 3層
clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=[10, 10])  # 4層

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(clf.score(X_test, y_test))
