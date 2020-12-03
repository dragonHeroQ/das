import numpy as np
import os.path as osp
from scipy import sparse
from sklearn.model_selection import train_test_split


def load_airfoil():

    base_dir = osp.dirname(osp.abspath(__file__))
    data = []
    with open(osp.join(base_dir, "airfoil_self_noise.dat"), 'r') as f:
        for line in f:
            tmp = [float(x) for x in line.split()]
            data.append(tmp)

    data = np.asarray(data)
    data = np.nan_to_num(data, copy=False)

    data_X = data[:, :-1]
    data_y = data[:, -1].ravel()

    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


def load_airfoil_sparse():
    data = []
    with open("airfoil_self_noise.dat", 'r') as f:
        for line in f:
            tmp = [float(x) for x in line.split()]
            data.append(tmp)

    data = np.asarray(data)
    data = np.nan_to_num(data, copy=False)

    data_X = data[:, :-1]
    data_y = data[:, -1].ravel()

    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)
    return sparse.csr_matrix(x_train), sparse.csr_matrix(x_test), y_train, y_test


if __name__ == '__main__':
    from benchmarks.data.benchmark_helper import getmbof
    X_train, X_test, Y_train, Y_test = load_airfoil()
    print("X_train, X_test = {}, {}".format(getmbof(X_train), getmbof(X_test)))
    # X_train, X_test = 0.04MB, 0.02MB
    print(np.min(Y_train), np.max(Y_train))
    print(np.min(Y_test), np.max(Y_test))
