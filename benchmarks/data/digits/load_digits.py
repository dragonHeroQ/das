import sklearn.datasets
from sklearn.model_selection import train_test_split


def load_digits():
	X, y = sklearn.datasets.load_digits(return_X_y=True)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_digits()
	from benchmarks.data.benchmark_helper import getmbof
	print(getmbof(X_train))
	# 0.59MB
