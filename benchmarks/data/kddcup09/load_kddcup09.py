import numpy as np
import os.path as osp
from sklearn.model_selection import train_test_split


def load_kddcup09():

    base_dir = osp.dirname(osp.abspath(__file__))
    X = np.load(osp.join(base_dir, "orange_X.npy"))
    y = np.load(osp.join(base_dir, "orange_y.npy"))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_kddcup09()
    print(X_train.shape)
    print(X_test.shape)
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(X_train), getmbof(Y_train))

