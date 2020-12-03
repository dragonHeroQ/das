import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import importlib
from automl.compmodel import CompositeModel
from automl.get_algorithm import get_algorithm_by_key
import pickle
import psutil
import multiprocessing
from multiprocessing import Process, Manager


def kill_tree(pid, including_parent=True):
	parent = psutil.Process(pid)
	for child in parent.children(recursive=True):
		print("child", child)
		child.kill()

	if including_parent:
		parent.kill()


if len(sys.argv) > 1:
	data_name = sys.argv[1]
else:
	data_name = 'mg'
print("Data Name: {}".format(data_name))
if data_name in ['abalone', 'cadata', 'space_ga']:
	data_dir = 'libsvm_{}'.format(data_name)
else:
	data_dir = data_name
load_xx = importlib.import_module('{}.load_{}'.format(data_dir, data_name))
x_train, x_test, y_train, y_test = getattr(load_xx, 'load_{}'.format(data_name))()
mse_dict = pickle.load(open('{}_mses.pkl'.format(data_name), 'rb'))
# print(mse_dict)
sorted_mses = sorted(mse_dict.items(), key=lambda x: x[1], reverse=False)
for k, v in sorted_mses[:10]:
	idx, algo1, algo2, cat_type = k.split('+')
	comp = CompositeModel([(algo1, get_algorithm_by_key(algo1, random_state=0)),
						   (algo2, get_algorithm_by_key(algo2, random_state=42)),
						   cat_type])
	evaluation_rule = 'mean_squared_error'
	manager = multiprocessing.Manager()
	d = manager.dict()
	p = Process(target=comp.score, args=(x_train, y_train, x_test, y_test, evaluation_rule), kwargs={"return_dict": d})
	p.start()
	p.join(360)
	if p.is_alive():
		kill_tree(p.pid)
	score = np.inf
	if 'test_score' in d.keys():
		score = d["test_score"]
	# score = comp.score(X=x_train, y=y_train, x_test=x_test, y_test=y_test, evaluation_rule=evaluation_rule)
	print(k, v[0], score, v[1])

