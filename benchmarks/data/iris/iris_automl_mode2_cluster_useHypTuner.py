import sys
sys.path.append('../../')
from automl.Clustering.cluster import Cluster
import sklearn.datasets
from sklearn.model_selection import train_test_split
from automl.performance_evaluation import eval_performance
from automl.HypTuner.iteration.iteration_datum import Datum
from automl.BaseAlgorithm.Preprocessing.SKLearnPreprocessing.MinMaxScaler import MinMaxScaler
from automl.BaseAlgorithm.FeatureEngineering.SKLearnFeatureEngineering.UnivariateFeatureSelection import SelectFpr
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.RandomForest import RandomForest
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.ExtraTreesClassifier import ExtraTreesClassifier
from automl.HypTuner.result import Result
from sklearn.externals import joblib
import time
import warnings
warnings.filterwarnings("ignore")

X, y = sklearn.datasets.load_iris(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(len(x_train), len(x_test))
# print(x_train.shape, y_train.shape)

PIP_VAL_RESULT = []
PIP_TEST_RESULT = []
VAL_RESULT = []
TEST_RESULT = []
TIME_COST = []
name = "Pipeline"
time_pipeline = 80
time_hyptuner = 60

evaluation_rule = "silhouette_score"

for i in range(1):
    start_time = time.time()

    clf = Cluster(total_timebudget=140, per_run_timebudget=60, automl_mode=2,
                  # evaluation_rule=evaluation_rule,
                  budget_type="iter", min_budget=27, max_budget=729, name=name,
                  use_HypTuner=True, time_budget_pipeline_HypTuner=1/1,
                  output_folder="./HypTuner_log/{}".format(name), verbose=False)

    clf.configure_default_algorithm_set()
    # clf.addDataPreprocessor({"MinMaxScalar": MinMaxScaler()})
    # clf.addFeatureSelector({"SelectFpr": SelectFpr()})
    # clf.addClassifier({"RandomForestClassifier": RandomForest()})
    # clf.addClassifier({"ExtraTreesClassifier": ExtraTreesClassifier()})
    # ['MinMaxScaler', 'SelectFpr', 'RandomForestClassifier']
    # clf._fit_based_on_q_learning(x_train, y_train, time_budget=time_pipeline)
    # print("fit ended!")
    # mod = joblib.load("model_%d_%d.pkl" % (clf.get_version(), clf.get_best_model()))
    # y_hat = mod.predict(x_test)
    # best_test_score = eval_performance(rule=evaluation_rule, y_true=y_test, y_score=y_hat)
    # print("Best Pipeline Test Score: ", best_test_score)
    # print("Best Validation Score: ", clf.get_best_val_score())
    # print("Best Pipeline: ", clf.get_best_model_name())
    # PIP_TEST_RESULT.append(best_test_score)
    # PIP_VAL_RESULT.append(clf.get_best_val_score())
    # a = clf.get_best_model_inst()
    # # a = get_pipeline_by_key(clf.get_best_model_name())
    # print('params: ', a.get_params())
    #
    # records = clf.get_config_performance_record_by_key(clf.get_best_model_name())
    # # print("RECORDS: {}".format(records))
    # iter_data = dict()
    # for i, config_score in enumerate(records):
    #     config = config_score[0]
    #     score = config_score[1]
    #     iter_data.update({(-1, 0, i): Datum.build_iteration(config, score, 729, evaluation_rule=evaluation_rule)})
    #     print("\nupdate {}: {}->{}".format((-1, 0, i), config, score))
    #
    # # iter_data.update({(-1, 0, 0): Datum.build_iteration(a.get_params(),
    # #                                                     clf.get_best_val_score(), 729)})
    # warm_start_result = Result(iter_data)
    # # print("WARM_START_RESULT: ", warm_start_result.get_id2config_mapping())
    # # print('warm_start: ', warm_start_result)
    # # a = get_pipeline_by_key(['QuadraticDA'])
    # clf.set_automl_mode(0)
    # clf.clearClassifier()
    # clf.addClassifier({'a': a})
    clf.fit(X, y, all_budgets=False)

    try:
        y_hat = clf.predict(X)
        acc = eval_performance(rule=evaluation_rule, X=X, y_true=y, y_score=y_hat)
        print("from pipeline searching {} to HypTuner {}: ".format(0, name), acc)
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
