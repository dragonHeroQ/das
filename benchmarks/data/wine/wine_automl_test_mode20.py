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
time_pipeline = 60
time_hyptuner = 60
classification_mode = 1
evaluation_rule = "accuracy_score"

for i in range(1):
    start_time = time.time()

    clf = Classifier(total_timebudget=time_hyptuner, per_run_timebudget=60,
                     automl_mode=0, classification_mode=classification_mode,
                     evaluation_rule=evaluation_rule,
                     validation_strategy="cv", validation_strategy_args=3, budget_type="datapoints",
                     min_budget=27, max_budget=729, name=name,
                     output_folder="./HypTuner_log/{}".format(name),
                     verbose=False)

    clf.set_version(i)
    clf.set_automl_mode(2)
    clf.addClassifier({"LinearRegression": LogisticRegression()})
    clf.addDataPreprocessor({"MaxAbsScalar": MaxAbsScaler()})
    clf.addFeatureSelector({"FactorAnalysis": FactorAnalysis()})
    # clf.configure_default_algorithm_set()
    clf._fit_based_on_q_learning(x_train, y_train, time_budget=time_pipeline)
    print("fit ended!")
    mod = joblib.load("model_%d_%d.pkl" % (clf.get_version(), clf.get_best_model()))
    y_hat = mod.predict(x_test)
    best_test_score = eval_performance(rule=evaluation_rule, y_true=y_test, y_score=y_hat)
    print("Best Pipeline Test Score: ", best_test_score)
    print("Best Validation Score: ", clf.get_best_val_score())
    print("Best Pipeline: ", clf.get_best_model_name())
    PIP_TEST_RESULT.append(best_test_score)
    PIP_VAL_RESULT.append(clf.get_best_val_score())
    a = clf.get_best_model_inst()
    print("type(a) = {}".format(type(a)))
    # a = get_pipeline_by_key(clf.get_best_model_name())
    print('params: ', a.get_params())

    records = clf.get_config_performance_record_by_key(clf.get_best_model_name())
    # print("RECORDS: {}".format(records))
    iter_data = dict()
    for i, config_score in enumerate(records):
        config = config_score[0]
        score = config_score[1]
        iter_data.update({(-1, 0, i): Datum.build_iteration(config, score, 729, evaluation_rule=evaluation_rule)})
        print("\nupdate {}: {}->{}".format((-1, 0, i), config, score))

    # iter_data.update({(-1, 0, 0): Datum.build_iteration(a.get_params(),
    #                                                     clf.get_best_val_score(), 729)})
    warm_start_result = Result(iter_data)
    # print('warm_start: ', warm_start_result)
    # a = get_pipeline_by_key(['FactorAnalysis', 'LogisticRegression'])
    # print(a.get_params())
    clf.set_automl_mode(0)
    clf.clearClassifier()
    clf.addClassifier({'a': a})
    clf.fit(x_train, y_train, warm_start_result=warm_start_result, verbose=False, all_budgets=False)

    try:
        y_hat = clf.predict(x_test)
        acc = eval_performance(rule=evaluation_rule, y_true=y_test, y_score=y_hat)
        print("from pipeline searching {} to HypTuner {}: ".format(best_test_score, name), acc)
        TEST_RESULT.append(acc)
    except Exception as e:
        print(e)
        TEST_RESULT.append(0.0)

    time_cost = time.time() - start_time
    print("Time_Cost: {}".format(time_cost))
    TIME_COST.append(time_cost)
    VAL_RESULT.append(clf.hypTunerResult.get_incumbent_val_score())
    res = clf.hypTunerResult
    incum_id = res.get_incumbent_id(all_budgets=False)
    print("INCUM RUN: ", res.get_runs_by_id(incum_id))


print("Pipeline           val_results: ", PIP_VAL_RESULT)
print("Pipeline+HypTuner  val_results: ", VAL_RESULT)
print("Pipeline          test_results: ", PIP_TEST_RESULT)
print("Pipeline+HypTuner test_results: ", TEST_RESULT)

print("time costs: ", TIME_COST)
