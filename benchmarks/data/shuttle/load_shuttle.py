import numpy as np
import os.path as osp


def load_shuttle(return_X_y=False):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    base_dir = osp.dirname(osp.abspath(__file__))
    with open(osp.join(base_dir, "shuttle.trn"), 'r') as f:
        for line in f:
            tmp = [int(x) for x in line.split(" ")]
            X_train.append(tmp[:-1])
            Y_train.append(int(tmp[-1]))

    with open(osp.join(base_dir, "shuttle.tst.txt"), 'r') as f:
        for line in f:
            tmp = [int(x) for x in line.split(" ")]
            X_test.append(tmp[:-1])
            Y_test.append(tmp[-1])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    if return_X_y is False:
        return X_train, X_test, Y_train, Y_test
    else:
        return np.vstack([X_train, X_test]), np.hstack([Y_train, Y_test])


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_shuttle()
    print(x_train.shape)
    print(x_test.shape)
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(x_train), getmbof(y_train))
    # print(np.unique(y_train))

