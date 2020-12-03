import sys
import os
import psutil
import numpy as np


def get_regexp_dict_value(dict_, key_):
	for key in dict_:
		if key_ in key:
			return dict_[key]
	return None


def getmbof(x):
	if isinstance(x, np.ndarray):
		return x.itemsize * x.size / 1048576.0
	return sys.getsizeof(x) / 1048576.0


def kill_tree(pid, including_parent=True):
	parent = psutil.Process(pid)
	for child in parent.children(recursive=True):
		print("child", child)
		child.kill()

	if including_parent:
		parent.kill()


def search_newest_from_dir(directory):
	if directory is None:
		return None
	file_lists = os.listdir(directory)
	if not file_lists:
		return None
	file_lists = map(lambda x: (int(x.split('.')[0]), x), file_lists)
	sorted_file_lists = sorted(file_lists, key=lambda x: x[0], reverse=True)
	return sorted_file_lists[0][1]


def min_max_scale(dic: dict=None):
	min_val = min(dic.values())
	max_val = max(dic.values())
	res = {}
	if min_val == max_val:
		for i in dic.keys():
			res[i] = 1 / len(dic.keys())
	else:
		for i in dic.keys():
			res[i] = (max_val - dic[i]) / (max_val - min_val)
	return res


def soft_max(s: dict):
	tmp_arr = list(s.values())
	tmp_arr = np.array(tmp_arr)
	tmp_arr = np.nan_to_num(tmp_arr)
	pi = {}
	for i in s.keys():
		pi[i] = np.exp(s[i]) / (np.sum(np.exp(tmp_arr)))
	return pi


def flip_coin(p):
	"""
	If return True, go on, else stop.
	:param p:
	:return:
	"""
	# p is the probability to go on
	return np.random.uniform() <= p
