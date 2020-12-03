import sys
sys.path.append("../")
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from airfoil_self_noise.load_airfoil import load_airfoil

x_train, x_test, y_train, y_test = load_airfoil()

print(y_train)

cm1 = ExtraTreesClassifier()
cm1.fit(x_train, y_train)

y_train_proba = cm1.predict_proba(x_train)
y_test_proba = cm1.predict_proba(x_test)

cm2 = ExtraTreesRegressor()
cm2.fit(y_train_proba, y_train)
y_hat = cm2.predict(x_test)

mse = mean_squared_error(y_test, y_hat)
print("CompModel on airfoil: {}".format(mse))



