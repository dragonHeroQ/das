import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp


def load_superconduct():
    base_dir = osp.dirname(osp.abspath(__file__))
    df = pd.read_csv(osp.join(base_dir, "train.csv"), sep=",")
    data = df.values
    data = np.nan_to_num(data, copy=False)

    data_X = data[:, :-1]
    data_y = data[:, -1].ravel()

    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_superconduct()
    print(x_train.shape, y_train.shape)
    from benchmarks.data.benchmark_helper import getmbof
    print(getmbof(x_train), getmbof(x_test))
    # 8.80MB 4.34MB
