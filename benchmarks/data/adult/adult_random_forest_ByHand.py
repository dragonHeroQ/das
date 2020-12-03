import sys
sys.path.append('../../')
from automl.Classification.classifier import Classifier
import sklearn.datasets
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm import SVM, RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
import time
import warnings
from uci_adult import load_data
warnings.filterwarnings("ignore")

# X, y = sklearn.datasets.load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = load_data()
print(len(x_train), len(x_test))

RESULT = []


for i in range(1):
    start_time = time.time()

    estimator = RandomForest.RandomForest()

    estimator.fit(x_train, y_train)

    y_hat = estimator.predict(x_test)

    time_cost = time.time() - start_time
    acc = accuracy_score(y_test, y_hat)
    print("RandomForest by Hand on ADULT: ", acc)
    print("Time_Cost: {}".format(time_cost))
    RESULT.append((acc, time_cost))

    # 0.8466924636078865
