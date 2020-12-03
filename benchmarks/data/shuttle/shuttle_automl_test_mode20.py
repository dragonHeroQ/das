import sys
sys.path.append('../../')
from automl.Clustering.cluster import Cluster
import sklearn.datasets
from sklearn.model_selection import train_test_split
from automl.performance_evaluation import eval_performance
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
total_time = 3600

evaluation_rule = "silhouette_score"

for i in range(1):
    start_time = time.time()

    clf = Cluster(total_timebudget=total_time, per_run_timebudget=60, automl_mode=2,
                  evaluation_rule=evaluation_rule,
                  budget_type="iter", min_budget=27, max_budget=729, name=name,
                  use_HypTuner=True, time_budget_pipeline_HypTuner=2/1,
                  output_folder="./MODE20_log/{}".format(name), verbose=False)

    clf.configure_default_algorithm_set()
    clf.fit(X, y)

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

print("Pipeline+HypTuner  val_results: ", VAL_RESULT)
print("Pipeline+HypTuner test_results: ", TEST_RESULT)
print("time costs: ", TIME_COST)
