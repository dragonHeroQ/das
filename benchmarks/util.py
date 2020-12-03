import time
import importlib
import numpy as np
import sys
import socket

HOSTNAME = socket.gethostname()


def getmbof(x):
	if isinstance(x, np.ndarray):
		return "{:.2f}MB".format(x.itemsize * x.size / 1048576.0)
	return "{:.2f}MB".format(sys.getsizeof(x) / 1048576.0)


def getMBSize(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1048576.0
	return sys.getsizeof(x) / 1048576.0


def getKBSize(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1024.0
	return sys.getsizeof(x) / 1024.0


def get_total_MBSize(x, x_):
	total_memory = (getMBSize(x) + getMBSize(x_))
	if total_memory < 1.0:
		return 1
	elif total_memory < 40.0:
		return 1.5
	elif total_memory < 100.0:
		return 2
	elif total_memory < 300.0:
		return 3
	elif total_memory < 600.0:
		return 4
	elif total_memory < 800.0:
		return 5
	elif total_memory < 1000.0:
		return 6
	else:
		return 10


CLASSIFICATION = ['adult', 'letter', 'digits', 'wine', 'iris',
                  'breast_cancer', 'dexter', 'gisette', 'imdb',
                  'yeast', 'kddcup09', 'shuttle', 'lungcancer']

REGRESSION = ['mg', 'airfoil', 'space_ga', 'abalone',
              'cadata', 'superconduct', 'cpusmall']


def load_data(data_name):
	if data_name in ['abalone', 'cadata', 'space_ga']:
		data_dir = 'libsvm_{}'.format(data_name)
	else:
		data_dir = data_name
	load_xx = importlib.import_module('data.{}.load_{}'.format(data_dir, data_name))
	return getattr(load_xx, 'load_{}'.format(data_name))()


def strftime(t=None):
	return time.strftime("%m%d%H%M", time.localtime(t or time.time()))


def convert_value_type(value_type: str, value: str):
	assert value_type in ['str', 'int', 'auto'], "Unknown value_type={}".format(value_type)
	if value_type == 'str':
		return str
	if value_type == 'int':
		return int
	for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
		if ch in value:
			return str
	return int


def extract_key_value(key_value_str: str, value_type='int'):
	# key_value_str: 'key=value'
	key_value_str = str(key_value_str)
	key = str(key_value_str.split('=')[0])
	value = key_value_str.split('=')[1]
	value = convert_value_type(value_type=value_type, value=value)(value)
	return key, value


def extract_specification(argvs, search_algo=None):
	"""
	Command Format: python <search_algo>.py <data_name> rng=<rng> time=<time> [optional]TL=<run_time_limit>

	:param argvs:
	:param search_algo:
	:return:
	"""
	assert search_algo is not None, "You must provide search_algo!"
	# print("argvs = {}".format(argvs))
	data_name = 'digits'
	if len(argvs) > 1:
		data_name = argvs[1]

	x_train, x_test, y_train, y_test = load_data(data_name)

	if data_name in CLASSIFICATION:
		n_classes = len(np.unique(y_train))
		evaluation_rule = 'accuracy_score'
	else:
		evaluation_rule = 'r2_score'
		n_classes = 1

	rng_str = "rng=0"
	if len(argvs) > 2:
		rng_str = str(argvs[2]).lower()
	rng_key, rng_value = extract_key_value(rng_str, value_type='int')
	assert rng_key == 'rng', "please use rng, instead of {}".format(rng_key)

	budget = "time=1200.0"
	if len(argvs) > 3:
		budget = str(argvs[3]).lower()
	budget_type, total_budget = extract_key_value(budget, value_type='int')

	return_dict = {}
	plus_info = []
	for argv_str in argvs[4:]:
		key, value = extract_key_value(argv_str, value_type='auto')
		return_dict[key] = value
		plus_info.append((key, value))

	plus_info = ["{}_{}".format(key, value) for key, value in plus_info]
	plus_info_str = "_".join(plus_info)
	if len(plus_info_str) > 0:
		plus_info_str += "_"
	if 'TL' not in plus_info_str:
		plus_info_str = '{}_{}_'.format('TL', 240) + plus_info_str

	time_str = strftime()
	lcv_f_name = "{hostname}_{search_algo}_{data_name}_r{rng}_{budget_type}" \
	             "_{total_budget}_{plus_info_str}{time_str}.lcv".format(hostname=HOSTNAME, search_algo=search_algo,
	                                                                    data_name=data_name, rng=rng_value,
	                                                                    budget_type=budget_type,
	                                                                    plus_info_str=plus_info_str,
	                                                                    total_budget=total_budget, time_str=time_str)

	return_dict.update({'dataset_tuple': (x_train, x_test, y_train, y_test),
	                    rng_key: rng_value, 'total_budget': total_budget, 'budget_type': budget_type,
	                    'n_classes': n_classes, 'evaluation_rule': evaluation_rule, 'lcv_f_name': lcv_f_name,
	                    'data_name': data_name, 'hostname': HOSTNAME})

	return return_dict


def experiment_summary(return_dict: dict):
	print("=================== EXPERIMENT SUMMARY ==============================")
	print("Host Name: {}".format(HOSTNAME))
	print("DataSet: {}, Train Shape = {}, Train Size = {}".format(return_dict['data_name'],
	                                                              return_dict['dataset_tuple'][0].shape,
	                                                              getmbof(return_dict['dataset_tuple'][0])))
	print("RNG: {}, Total Budget: {} = {}".format(return_dict['rng'],
	                                              return_dict['budget_type'],
	                                              return_dict['total_budget']))
	print("n_classes = {}, evaluation_rule = {}\nStoring learning curve to {}".format(return_dict['n_classes'],
	                                                                                  return_dict['evaluation_rule'],
	                                                                                  return_dict['lcv_f_name']))
	for key in return_dict.keys():
		if key not in ['data_name', 'dataset_tuple', 'total_budget', 'rng', 'hostname',
		               'budget_type', 'n_classes', 'evaluation_rule', 'lcv_f_name']:
			print("{} = {}".format(key, return_dict[key]))
	print("=====================================================================")


if __name__ == '__main__':
	# X_train, X_test, Y_train, Y_test = load_data('imdb')
	# print(X_train.shape, X_test.shape)
	# print(Y_train.shape, Y_test.shape)
	print(extract_key_value("nbandit=10", 'auto'))
	# return_dict = extract_specification(['ucb.py', 'digits', 'rng=0', 'time=200', 'n_bandit=15'], 'ucb')
	# print(return_dict.keys())
