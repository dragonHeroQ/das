import sys
sys.path.append('../../')
from automl.Classification.classifier import Classifier
import sklearn.datasets
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.LogisticRegression import *
from automl.BaseAlgorithm.Preprocessing.SKLearnPreprocessing.MaxAbsScaler import *
from automl.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering.FactorAnalysis import *
from sklearn.model_selection import train_test_split
from automl.performance_evaluation import eval_performance
from automl.HypTuner.iteration.iteration_datum import Datum
from automl.get_algorithm import get_pipeline_by_key
from automl.HypTuner.result import Result
from sklearn.externals import joblib
import time
import warnings
warnings.filterwarnings("ignore")

X, y = sklearn.datasets.load_wine(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(x_train.shape, x_test.shape)

PIP_VAL_RESULT = []
PIP_TEST_RESULT = []
VAL_RESULT = []
TEST_RESULT = []
TIME_COST = []
name = "Pipeline"
time_time = 3600
classification_mode = 1
evaluation_rule = "accuracy_score"

for i in range(10):
    start_time = time.time()

    clf = Classifier(total_timebudget=time_time, per_run_timebudget=60,
                     automl_mode=2, classification_mode=classification_mode,
                     evaluation_rule=evaluation_rule,
                     use_HypTuner=True, time_budget_pipeline_HypTuner=2/1,
                     validation_strategy="cv", validation_strategy_args=3, budget_type="datapoints",
                     min_budget=27, max_budget=729, name=name,
                     output_folder="./MODE20_log/{}".format(name),
                     verbose=False, version=i)

    clf.configure_default_algorithm_set()

    clf.fit(x_train, y_train)

    try:
        y_hat = clf.predict(x_test)
        acc = eval_performance(rule=evaluation_rule, y_true=y_test, y_score=y_hat)
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
