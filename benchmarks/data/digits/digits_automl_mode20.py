import sys
sys.path.append('../../')
from automl.Classification.classifier import Classifier
import sklearn.datasets
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm import SVM
from sklearn.model_selection import train_test_split
from automl.get_algorithm import *
from automl.performance_evaluation import eval_performance
from automl.HypTuner.iteration.iteration_datum import Datum
from automl.HypTuner.result import Result
from sklearn.externals import joblib
import time
import warnings
warnings.filterwarnings("ignore")

X, y = sklearn.datasets.load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(x_train), len(x_test))

PIP_VAL_RESULT = []
PIP_TEST_RESULT = []
VAL_RESULT = []
TEST_RESULT = []
TIME_COST = []
name = "Pipeline"
time_time = 60
classification_mode = 1
evaluation_rule = "top_1_accuracy"

for i in range(1):
    start_time = time.time()

    clf = Classifier(total_timebudget=time_time, per_run_timebudget=60,
                     automl_mode=2, classification_mode=classification_mode,
                     evaluation_rule=evaluation_rule, predict_method="predict_proba",
                     use_HypTuner=True, time_budget_pipeline_HypTuner=2/1,
                     validation_strategy="cv", validation_strategy_args=3, budget_type="datapoints",
                     min_budget=27, max_budget=729, name=name,
                     output_folder="./MODE20_log/{}".format(name),
                     verbose=False, version=i)

    clf.configure_default_algorithm_set()
    clf.fit(x_train, y_train)

    try:
        y_hat = clf.predict(x_test)
        acc = eval_performance(evaluation_rule, y_test, y_hat)
        print("from pipeline searching {} to HypTuner {}: ".format(0, name), acc)
        TEST_RESULT.append(acc)
    except Exception as e:
        print(e)
        TEST_RESULT.append(0.0)

    time_cost = time.time() - start_time
    print("Time_Cost: {}".format(time_cost))
    TIME_COST.append(time_cost)
    VAL_RESULT.append(clf.hypTunerResult.get_incumbent_val_score())

print("Pipeline+HypTuner  val_results: ", VAL_RESULT)
print("Pipeline+HypTuner test_results: ", TEST_RESULT)
print("time costs: ", TIME_COST)
