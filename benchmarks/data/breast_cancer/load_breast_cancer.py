import sklearn.datasets
from sklearn.model_selection import train_test_split


def load_breast_cancer():
	X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	x_train, _, _, _ = load_breast_cancer()
	from benchmarks.data.benchmark_helper import getmbof
	print(getmbof(x_train))
