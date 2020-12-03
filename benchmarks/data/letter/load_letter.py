import numpy as np
import os.path as osp


def load_letter():
    """
    Load UCI LETTER data, if not exists in data_base_dir, download it and put into data_base_dir.

    :return: x_train, y_train, x_test, y_test
    """
    base_dir = osp.dirname(osp.abspath(__file__))
    data_path = osp.join(base_dir, "./letter-recognition.data")
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X = np.zeros((n_datas, 16), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X[i, :] = list(map(float, row[1:]))
        y[i] = ord(row[0]) - ord('A')
    x_train, y_train = X[:16000], y[:16000]
    x_test, y_test = X[16000:], y[16000:]
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_letter()
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(X_train))
    # 0.98MB
