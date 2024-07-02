from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)

print(any(dummy_clf.predict(X_train))) # prints False: no 5s detected
print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))