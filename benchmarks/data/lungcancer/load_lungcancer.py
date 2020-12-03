import numpy as np
import pandas as pd
import os.path as osp
from sklearn.model_selection import train_test_split


def load_lungcancer():

	base_dir = osp.dirname(osp.abspath(__file__))
	data = pd.read_csv(osp.join(base_dir, "lung-cancer.data.txt"))
	data = data.replace('?', 0)
	data = np.array(data)
	X = data[:, 1:]
	y = data[:, 0]
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_lungcancer()
	print(X_train.shape)
	print(X_test.shape)
	from benchmarks.data.benchmark_helper import getmbof
	print(getmbof(X_train), getmbof(Y_train))

