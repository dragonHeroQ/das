import numpy as np
import os.path as osp


def load_dexter():

    base_dir = osp.dirname(osp.abspath(__file__))

    x_train = np.load(osp.join(base_dir, "dexter_x_train.npy"))
    x_test = np.load(osp.join(base_dir, "dexter_x_test.npy"))
    y_train = np.load(osp.join(base_dir, "dexter_y_train.npy"))
    y_test = np.load(osp.join(base_dir, "dexter_y_test.npy"))

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_dexter()
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(X_train), getmbof(Y_train))
    print(X_train.shape, X_test.shape, Y_train.shape)
    # (300, 20000) (300, 20000)
    # 45.78MB
