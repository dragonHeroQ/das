import sys
sys.path.append("../")
sys.path.append("../../")
import os
import das
import logging
import importlib
from das.ArchitectureSearch.Evaluator.BaseEvaluator import BaseEvaluator

logger = logging.getLogger(das.logger_name)
util = importlib.import_module('util')
extract_key_value = getattr(util, 'extract_key_value')

CLASSIFICATION = ['adult', 'letter', 'digits', 'wine', 'iris', 'breast_cancer', 'dexter', 'gisette', 'imdb', 'yeast']
REGRESSION = ['mg', 'airfoil', 'space_ga', 'abalone', 'cadata', 'superconduct', 'cpusmall']

LCV_DIR = "./exp_lcvs"


def extract_file_with_latest_time_str(match_str):
	# print("match_str", match_str)
	files = os.listdir(os.path.join(os.path.abspath(__file__), LCV_DIR))
	matched_files = []
	for file_name in files:
		if match_str in file_name:
			matched_files.append(file_name[:-4].split('_'))
	assert len(matched_files) > 0, "No matched files for {}".format(match_str)
	sorted_matched_files = sorted(matched_files, key=lambda x: x[-1], reverse=True)
	# print(sorted_matched_files[0])
	return sorted_matched_files[0]


def extract_information_from_filename(filename):
	parts = filename.split('_')
	info = dict()
	info['hostname'] = parts[0]
	info['search_algos'] = parts[1]
	info['data_name'] = parts[2]
	info['rng'] = parts[3][1:]
	info['time_str'] = parts[-1]
	for i in range(4, len(parts)-1, 2):
		key = parts[i]
		value = parts[i+1]
		info[key] = value

	return info


def show_learning_curve(hostname='WHATBEG',
                        search_algos: list = None,
                        data_name='digits',
                        rng=0,
                        budget_type='time',
                        total_budget=200,
                        other_info_str=None,
                        time_str='latest'):
	learning_curves = []
	plus_info = []
	if other_info_str:
		for argv_str in other_info_str.split(","):
			key, value = extract_key_value(argv_str, value_type='auto')
			plus_info.append((key, value))

	plus_info = ["{}_{}".format(key, value) for key, value in plus_info]
	plus_info_str = "_".join(plus_info)
	if len(plus_info_str) > 0:
		plus_info_str += "_"
	if 'TL' not in plus_info_str:
		plus_info_str = '{}_{}_'.format('TL', 240) + plus_info_str

	for search_algo in search_algos:
		match_str = "{hostname}_{search_algo}_{data_name}_r{rng}_{budget_type}" \
		            "_{total_budget}_{plus_info_str}".format(
			hostname=hostname, search_algo=search_algo, data_name=data_name, rng=rng,
			budget_type=budget_type, total_budget=total_budget, plus_info_str=plus_info_str)
		file_components = extract_file_with_latest_time_str(match_str=match_str)
		time_string = time_str
		if time_str == 'latest':
			time_string = file_components[-1]
		hostname_string = hostname
		if not hostname:
			hostname_string = file_components[0]
		lcv_f_name = "{hostname}_{search_algo}_{data_name}_r{rng}_{budget_type}" \
		             "_{total_budget}_{plus_info_str}{time_str}.lcv".format(
			hostname=hostname_string, search_algo=search_algo, data_name=data_name, rng=rng,
			budget_type=budget_type, total_budget=total_budget, plus_info_str=plus_info_str,
			time_str=time_string
		)
		# lcv_f_name = "{match_str}{time_str}.lcv".format(match_str=match_str, time_str=time_string)
		logger.debug("LCV f_name = {}".format(lcv_f_name))
		learning_curve = BaseEvaluator.load_learning_curve(lcv_f_name,
		                                                   base_dir=os.path.join(os.path.dirname(__file__), LCV_DIR))
		learning_curves.append(('{search_algo}'.format(search_algo=search_algo), learning_curve))

	title = "{data_name} (rng={rng}, {budget_type}={total_budget}, {plus_info_str})".format(
		data_name=data_name, rng=rng, budget_type=budget_type, total_budget=total_budget,
		plus_info_str=plus_info_str)
	BaseEvaluator.plot_learning_curves(title=title,
	                                   learning_curves=learning_curves,
	                                   time_with='val_r2_score')


def show_learning_curve_v2(hostname='slave',
	                       search_algos='*',
	                       data_name='digits',
	                       rng='*',
	                       budget_type='*',
	                       total_budget='*',
	                       other_info_str=None,
	                       time_str='*',
                           **kwargs):
	learning_curves = []
	plus_info = {}
	if other_info_str:
		for argv_str in other_info_str.split(","):
			key, value = extract_key_value(argv_str, value_type='auto')
			plus_info[key] = value

	if data_name in CLASSIFICATION:
		time_with = 'val_accuracy_score'
	else:
		time_with = 'val_r2_score'

	for file_name in os.listdir(os.path.join(os.path.dirname(__file__), LCV_DIR)):
		# print(file_name)
		if ".lcv" not in file_name:
			continue
		info = extract_information_from_filename(file_name)
		if (hostname == '*' or hostname in info['hostname'])\
			and (search_algos == '*' or search_algos in info['search_algos'])\
			and (data_name == '*' or data_name == info['data_name'])\
			and (rng == '*' or rng == info['rng'])\
			and (budget_type == '*' or budget_type == info['budget_type'])\
			and (total_budget == '*' or total_budget == info['total_budget'])\
			and (time_str == '*' or time_str == info['time_str']):

			learning_curve = BaseEvaluator.load_learning_curve(file_name,
			                                                   base_dir=os.path.join(os.path.dirname(__file__), LCV_DIR))
			time, objective = BaseEvaluator.extract_time_objective_from_lcv(learning_curve, time_with)
			learning_curves.append(("{}: {:.4f}".format(file_name[:-4], objective[-1]), learning_curve))

	title = data_name

	BaseEvaluator.plot_learning_curves(title=title,
	                                   learning_curves=learning_curves,
	                                   time_with=time_with,
	                                   **kwargs)


if __name__ == '__main__':
	# show_learning_curve(hostname="",
	#                     search_algos=['bo'],
	#                     data_name='airfoil',
	#                     rng=0,
	#                     budget_type='time',
	#                     total_budget=18000,
	#                     other_info_str=None,
	#                     time_str='latest')
	show_learning_curve_v2(data_name='digits')

