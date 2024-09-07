from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
X, y = mnist.data, mnist.target

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

some_digit = X[0]
# print("The prediction result : ", sgd_clf.predict([some_digit]))
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

print("The Threshold == 0")
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

print("The Threshold == 3000")
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

