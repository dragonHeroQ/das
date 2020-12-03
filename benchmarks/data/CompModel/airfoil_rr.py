import sys
sys.path.append("../")
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from airfoil.load_airfoil import load_airfoil
from cpusmall.load_cpusmall import load_cpusmall
from libsvm_cadata.load_cadata import load_cadata
x_train, x_test, y_train, y_test = load_cadata()

# ===========================================
# cm1 = ExtraTreesRegressor(random_state=None)
# cm1.fit(x_train, y_train)
#
# y_train_ba = cm1.predict(x_train)[:, np.newaxis]
# y_test_ba = cm1.predict(x_test)[:, np.newaxis]
#
# print(x_train.shape, y_train_ba.shape)
#
# aug_x_train = np.hstack((x_train, y_train_ba))
# aug_x_test = np.hstack((x_test, y_test_ba))
#
# cm2 = ExtraTreesRegressor(random_state=0)
# cm2.fit(aug_x_train, y_train)
# y_hat = cm2.predict(aug_x_test)
#
# mse = mean_squared_error(y_test, y_hat)
# print("CompModel on airfoil: {}".format(mse))

# 最高 2.716751680524193
# ===============================================
# cm1 = RandomForestRegressor(random_state=None)
# cm1.fit(x_train, y_train)
#
# y_train_ba = cm1.predict(x_train)[:, np.newaxis]
# y_test_ba = cm1.predict(x_test)[:, np.newaxis]
#
# print(x_train.shape, y_train_ba.shape)
#
# aug_x_train = np.hstack((x_train, y_train_ba))
# aug_x_test = np.hstack((x_test, y_test_ba))
#
# cm2 = RandomForestRegressor(random_state=0)
# cm2.fit(aug_x_train, y_train)
# y_hat = cm2.predict(aug_x_test)
#
# mse = mean_squared_error(y_test, y_hat)
# print("CompModel on airfoil: {}".format(mse))
# 最高 3.7350481504032254
# ===============================================
cm1 = ExtraTreesRegressor(random_state=0)
cm1.fit(x_train, y_train)

y_train_ba = cm1.predict(x_train)[:, np.newaxis]
y_test_ba = cm1.predict(x_test)[:, np.newaxis]

print(x_train.shape, y_train_ba.shape)

concat_type = 't'

if concat_type == 'c':
	aug_x_train = np.hstack((x_train, y_train_ba))
	aug_x_test = np.hstack((x_test, y_test_ba))
elif concat_type == 't':
	aug_x_train = y_train_ba
	aug_x_test = y_test_ba
else:
	raise NotImplementedError

cm2 = ExtraTreesRegressor(random_state=0)
cm2.fit(aug_x_train, y_train)
y_hat = cm2.predict(aug_x_test)

mse = mean_squared_error(y_test, y_hat)
print("CompModel on airfoil: {}".format(mse))
# 3.2392416905645116
# ===============================================
