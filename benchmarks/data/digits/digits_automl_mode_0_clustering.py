import sys
sys.path.append('../../')
import sklearn.datasets
from automl.Clustering.cluster import Cluster
from automl.performance_evaluation import eval_performance
from automl.BaseAlgorithm.Clustering.SKLearnBaseAlgorithm import KMeans

iris = []

X, y = sklearn.datasets.load_digits(return_X_y=True)

clt = Cluster(total_timebudget=60, per_run_timebudget=15, budget_type="iter",
              automl_mode=0, min_budget=3, max_budget=243, name="KMeans_DIGITS",
              output_folder="./HypTuner_log/KMeans_DIGITS")

clt.addClusteringAlgorithm({'KMeans': KMeans.KMeans()})

clt.fit(X)
y_hat = clt.predict(X)

print("aaaaaa", eval_performance("silhouette_score", X=X, y_true=y, y_score=y_hat))
# os.system("rm model_*")
iris.append(eval_performance("silhouette_score", X=X, y_true=y, y_score=y_hat))

k_clt = KMeans.KMeans(n_clusters=10)

y_hat = k_clt.fit_predict(X)
print("silhouette_score", eval_performance("silhouette_score", X=X, y_true=y, y_score=y_hat))

# TODO: 将budget作为max_iter
