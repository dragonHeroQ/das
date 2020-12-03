import os
from sklearn.externals import joblib


def print_model(filename):

    if not os.path.exists(filename):
        raise Exception("no such file %s" % filename)
    else:
        mod = joblib.load(filename)
        return mod.print_model()


if __name__ == "__main__":
    filename = "../../benchmarks/kddcup09/model_0_12.pkl"
    # filename = os.path.abspath(filename)
    print_model(filename)
