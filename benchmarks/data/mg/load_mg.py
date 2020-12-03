from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp


def load_mg():
    base_dir = osp.dirname(osp.abspath(__file__))
    data = load_svmlight_file(osp.join(base_dir, "mg.txt"))
    X = data[0]
    y = data[1]
    X = X.toarray()
    X = np.nan_to_num(X, copy=False)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    from benchmarks.data.benchmark_helper import getmbof
    x_train, x_test, y_train, y_test = load_mg()
    print("X_train, X_test = {}, {}".format(getmbof(x_train), getmbof(x_test)))
    # X_train, X_test = 0.04MB, 0.02MB
