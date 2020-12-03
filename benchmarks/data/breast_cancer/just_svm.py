import sys
sys.path.append('../../')
from automl.BaseAlgorithm.classification.SKLearnBaseAlgorithm.SVM import SVC
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import cross_val_score

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=None)
print(len(x_train), len(x_test))

print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)

svc = SVC()

svc.set_configuration_space()
svc.parameter_space.remove_hyperparameter("probability")

cs = svc.get_config_space()

config = cs.sample_configuration()

estimator = svc.new_estimator(config)

print(config.get_dictionary())
# res = svc.compute(config_id='sl', config=config.get_dictionary(),
#                   budget=6561, X=x_train, y=y_train)

result = cross_val_score(estimator, X, y, cv=3)

print(result)
