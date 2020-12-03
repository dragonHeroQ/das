import sys
sys.path.append("../")
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Airfoil ==============================================
from airfoil.load_airfoil import load_airfoil
from cpusmall.load_cpusmall import load_cpusmall
x_train, x_test, y_train, y_test = load_cpusmall()
m = ExtraTreesRegressor(random_state=0)
m.fit(x_train, y_train)
y_hat = m.predict(x_test)
mse = mean_squared_error(y_test, y_hat)
print("Baseline on airfoil: {}".format(mse))
# Baseline on airfoil: 2.7399956362499998

m = GradientBoostingRegressor(random_state=0)
m.fit(x_train, y_train)
y_hat = m.predict(x_test)
mse = mean_squared_error(y_test, y_hat)
print("Baseline on airfoil: {}".format(mse))
# Baseline on airfoil: 8.370813019825361

m = RandomForestRegressor(random_state=0)
m.fit(x_train, y_train)
y_hat = m.predict(x_test)
mse = mean_squared_error(y_test, y_hat)
print("Baseline on airfoil: {}".format(mse))
# Baseline on airfoil: 4.058300999314514

m = XGBRegressor(random_state=0)
m.fit(x_train, y_train)
y_hat = m.predict(x_test)
mse = mean_squared_error(y_test, y_hat)
print("Baseline on airfoil: {}".format(mse))
# Baseline on airfoil: 10.109102718352863

# A ==============================================
