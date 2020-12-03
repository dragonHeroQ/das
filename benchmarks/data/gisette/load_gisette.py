import numpy as np
import os.path as osp


def load_gisette():

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    base_dir = osp.dirname(osp.abspath(__file__))

    with open(osp.join(base_dir, "gisette_train.data.txt"), 'r') as f:
        for line in f:
            tmp = [int(x) for x in line.split()]
            x_train.append(tmp)

    with open(osp.join(base_dir, "gisette_train.labels.txt"), 'r') as f:
        for line in f:
            y_train.append(int(line))

    with open(osp.join(base_dir, "gisette_valid.data.txt"), "r") as f:
        for line in f:
            tmp = [int(x) for x in line.split()]
            x_test.append(tmp)

    with open(osp.join(base_dir, "gisette_valid.labels.txt"), 'r') as f:
        for line in f:
            y_test.append(int(line))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_gisette()
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(X_train))
    print(X_train.shape, X_test.shape)
    # 228.88MB
