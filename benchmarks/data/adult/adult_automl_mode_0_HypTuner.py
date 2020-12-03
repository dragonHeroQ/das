import sys
sys.path.append('../../')
from automl.Classification.classifier import Classifier
import sklearn.datasets
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm import SVM, RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
import time
import warnings
from load_adult import load_adult
warnings.filterwarnings("ignore")

# X, y = sklearn.datasets.load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = load_adult()
print(len(x_train), len(x_test))

VAL_RESULT = []
TEST_RESULT = []
TIME_COST = []
name = "RF_ADULT"
estimator = RandomForest.RandomForest()

for i in range(1):
    start_time = time.time()

    clf = Classifier(total_timebudget=80, per_run_timebudget=60,
                     automl_mode=0, classification_mode=0,
                     validation_strategy="cv", validation_strategy_args=3,
                     budget_type="datapoints",
                     min_budget=27, max_budget=729, name=name,
                     output_folder="./HypTuner_log/{}".format(name),
                     log_level='INFO')

    clf.addClassifier({"".format(name): estimator})

    # clf._fit_based_on_q_learning(, y)
    clf.fit(x_train, y_train)

    try:
        y_hat = clf.predict(x_test)
        acc = accuracy_score(y_test, y_hat)
        print("HypTuner {}: ".format(name), acc)
        TEST_RESULT.append(acc)
    except Exception as e:
        print(e)
        TEST_RESULT.append(0.0)

    time_cost = time.time() - start_time
    print("Time_Cost: {}".format(time_cost))
    TIME_COST.append(time_cost)
    # print(clf.hypTunerResult)
    VAL_RESULT.append(clf.hypTunerResult.get_incumbent_val_score())
    clf.hypTunerResult.save_trajectory("./HypTuner_log/{}".format(name),
                                       include_test=True,
                                       estimator=estimator,
                                       X_train=x_train,
                                       y_train=y_train, X_test=x_test, y_test=y_test,
                                       per_run_timebudget=60)


print(VAL_RESULT)
print(TEST_RESULT)
print(TIME_COST)
