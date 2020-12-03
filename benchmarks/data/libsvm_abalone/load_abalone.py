from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp


def load_abalone():
    base_dir = osp.dirname(osp.abspath(__file__))
    data = load_svmlight_file(osp.join(base_dir, "abalone.dat"))
    X = data[0]
    y = data[1]

    X = X.toarray()
    X = np.nan_to_num(X, copy=False)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_abalone()
    print(x_train.shape, y_train.shape)
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(x_train), getmbof(x_test))
    # 0.17MB 0.08MB
