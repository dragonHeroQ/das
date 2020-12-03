import os
import time
import json
import copy
import threading
from functools import reduce
import matplotlib.pyplot as plt
from das.performance_evaluation import loss_to_score
from sklearn.metrics.classification import accuracy_score
from das.HypTuner.iteration.base_iteration import Datum
from das.HypTuner.config_gen.config_space import is_legal_config_item


class Run(object):
	"""
	Not a proper class, more a 'struct' to bundle important
	information about a particular run
	"""
	def __init__(self, config_id, config, budget, loss, info, time_stamps, error_logs):
		self.config_id = config_id
		self.config = config
		self.budget = budget
		self.error_logs = error_logs
		self.loss = loss
		self.info = info
		self.time_stamps = time_stamps

	def __repr__(self):
		return(
			"config_id: %s\n" % (self.config_id,) +
			"config: {}\n".format(self.config) +
			"budget: %f\n" % self.budget +
			"loss: %s\n" % self.loss +
			"time_stamps: {submitted} (submitted), {started} (started), {finished} (finished)\n".format(**self.time_stamps) +
			"info: %s" % self.info
		)

	def __getitem__(self, k):
		"""
		In case somebody wants to use it like a dictionary
		"""
		return getattr(self, k)


class Result(object):
	"""
	Object returned by the trial_scheduler.run

	This class offers a simple API to access the information from a trial_scheduler run.

	Parameters
	----------
	iteration_data
		this holds all the configs and results of this iteration
	sched_config
	"""
	def __init__(self, iteration_data, sched_config=None):
		self.data = iteration_data
		self.sched_config = sched_config
		if self.sched_config is None:
			self.sched_config = {}
		self._merge_results()
		self.fit_model = None

	def __repr__(self):
		return ("type(data) = {}, data = {}\n".format(type(self.data), self.data) +
		        "schedule_config = {}".format(self.sched_config))

	def __getitem__(self, k):
		return self.data[k]

	def get_incumbent_id_only_max_budget(self):
		"""
		Find the config_id of the incumbent.

		The incumbent here is the configuration with the smallest loss
		among all runs on the maximum budget! If no run finishes on the
		maximum budget, None is returned!
		"""
		tmp_list = []
		for k, v in self.data.items():
			try:
				# only things run for the max budget are considered
				res = v.results[self.sched_config['max_budget']]
				if res is not None:
					tmp_list.append((res['loss'], k))
			except KeyError as e:
				print(e)
			except Exception as e:
				raise e

		if len(tmp_list) > 0:
			return min(tmp_list)[1]
		return None

	def old_get_incumbent_id(self):
		"""
		Find the config_id of the incumbent.

		The incumbent here is the configuration with the smallest loss
		among all runs on the maximum budget or the largest existing budget!

		When all_budgets is True, we consider the largest existing budget.
		If none of them returns results, return None.

		When all_budgets is False, we consider only the maximum budget.
		If no run finishes on the maximum budget, None is returned!
		"""
		# if not all_budgets:
		# 	return self.get_incumbent_id_only_max_budget()
		budget_dict = {}
		for b in self.sched_config['budgets']:
			budget_dict[b] = []
		for k, v in self.data.items():
			# NOT(we add) only things run for the max budget are considered
			for b in self.sched_config['budgets'][::-1]:
				if b in v.results:
					res = v.results[b]
					if res is not None:
						budget_dict[b].append((res['loss'], k))
					break
		for b in self.sched_config['budgets'][::-1]:
			if len(budget_dict[b]) > 0:
				return [min(budget_dict[b])[1], ]
		return [None, ]

	def get_incumbent_ids(self, all_budgets=True):
		if all_budgets is False:
			return self.old_get_incumbent_id()
		_, _, config_ids = self.get_trajectory_data(increase=False,
		                                            all_budgets=all_budgets,
		                                            budget_bigger_is_better=False,
		                                            non_decreasing_budget_limit=False)
		config_ids = config_ids[::-1]
		return reduce(lambda x, y: x if y in x else x + [y], [[], ]+config_ids)

	def get_incumbent_id(self, all_budgets=True):
		config_ids = self.get_incumbent_ids(all_budgets=all_budgets)
		assert len(config_ids) > 0, "config_ids is Empty!"
		return config_ids[0]

	def get_incumbent_configs(self, all_budgets=True):
		config_ids = self.get_incumbent_ids(all_budgets=all_budgets)
		id2config = self.get_id2config_mapping()
		# print("id2config = {}".format(id2config))
		configs = [id2config[cid]['config'] for cid in config_ids]
		return configs

	def get_incumbent_config(self, all_budgets=True):
		configs = self.get_incumbent_configs(all_budgets=all_budgets)
		assert len(configs) > 0, "len(incumbent_configs) == 0, no good config!"
		return configs[0]

	def get_incumbent_trajectory(self, all_budgets=True, budget_bigger_is_better=True,
	                             non_decreasing_budget_limit=True):
		"""
		Returns the best configurations over time

		Parameters
		----------
		all_budgets: bool
			If set to true all runs (even those not with the largest budget) can be the incumbent.
			Otherwise, only full budget runs are considered
		budget_bigger_is_better: bool
			flag whether an evaluation on a larger budget is always considered better.
			If True, the incumbent might increase for the first evaluations on a bigger budget
		non_decreasing_budget_limit: bool
			flag whether the budget of a new incumbent should be at least as big as the one for
			the current incumbent.
		Returns
		-------
		dict:
			dictionary with all the config IDs, the times the runs
			finished, their respective budgets, and corresponding losses
		"""
		all_runs = self.get_all_runs(only_largest_budget=not all_budgets)

		return_dict = {'config_ids': [],
		               'times_finished': [],
		               'budgets': [],
		               'losses': [],
		               }

		if len(all_runs) == 0:
			print("没有找到好的模型参数, 尝试更换分类器或者增加时间预算")
			return return_dict
		if not all_budgets:
			all_runs = list(filter(lambda r: r.budget == self.sched_config['max_budget'], all_runs))

		all_runs.sort(key=lambda r: r.time_stamps['finished'])

		current_incumbent = float('inf')
		incumbent_budget = self.sched_config['min_budget']

		for r in all_runs:

			if r.loss is None:
				continue

			new_incumbent = False

			if budget_bigger_is_better and r.budget > incumbent_budget:
				new_incumbent = True

			if r.loss < current_incumbent:
				new_incumbent = True

			# the budget of a new incumbent should be at least as big as the one for
			# the current incumbent.
			if non_decreasing_budget_limit and r.budget < incumbent_budget:
				new_incumbent = False

			if new_incumbent:
				# print("r.val_score, r.config_id, r.budget: ", 1-r.loss, r.config_id, r.budget)
				current_incumbent = r.loss
				incumbent_budget = r.budget

				return_dict['config_ids'].append(r.config_id)
				return_dict['times_finished'].append(r.time_stamps['finished'])
				return_dict['budgets'].append(r.budget)
				return_dict['losses'].append(r.loss)

		if current_incumbent != all_runs[-1].loss:
			r = all_runs[-1]

			return_dict['config_ids'].append(return_dict['config_ids'][-1])
			return_dict['times_finished'].append(r.time_stamps['finished'])
			return_dict['budgets'].append(return_dict['budgets'][-1])
			return_dict['losses'].append(return_dict['losses'][-1])

		return return_dict

	def get_runs_by_id(self, config_id):
		"""
		Returns a list of runs for a given config id

		The runs are sorted by ascending budget, so '-1' will give
		the longest run for this config.
		"""
		# print("Getting runs by id : {}".format(config_id))
		d = self.data[config_id]
		# print("data[config_id] = {}".format(d))
		# print("d.results.keys() = {}".format(d.results.keys()))
		runs = []
		for b in d.results.keys():
			try:
				err_logs = d.exceptions.get(b, None)

				if d.results[b] is None:
					r = Run(config_id, d.config, b, None, None, d.time_stamps[b], err_logs)
				else:
					r = Run(config_id, d.config, b, d.results[b]['loss'], d.results[b]['info'], d.time_stamps[b], err_logs)
				runs.append(r)
			except Exception as e:
				raise e
		runs.sort(key=lambda r: r.budget)
		return runs

	def get_all_runs(self, only_largest_budget=False):
		"""
		Returns all runs performed

		Parameters
		----------
		only_largest_budget: boolean
			if True, only the largest budget for each configuration
			is returned. This makes sense if the runs are continued
			across budgets and the info field contains the information
			you care about. If False, all runs of a configuration
			are returned
		"""
		all_runs = []

		for k in self.data.keys():

			runs = self.get_runs_by_id(k)
			if len(runs) > 0:
				if only_largest_budget:
					all_runs.append(runs[-1])
				else:
					all_runs.extend(runs)

		return all_runs

	def get_id2config_mapping(self):
		"""
		returns a dict where the keys are the config_ids and the values
		are the actual configurations
		"""
		new_dict = {}
		for k, v in self.data.items():
			new_dict[k] = {}
			new_dict[k]['config'] = copy.deepcopy(v.config)
			try:
				new_dict[k]['config_info'] = copy.deepcopy(v.config_info)
			except Exception as e:
				print(e)
		# print("MAPPING: {}".format(new_dict))
		return new_dict

	def _merge_results(self):
		"""
		Hidden function to merge the list of results into one
		dictionary and 'normalize' the time stamps
		"""
		if 'time_ref' in self.sched_config:
			new_dict = {}
			for it in self.data:
				new_dict.update(it)

			for k, v in new_dict.items():
				for kk, vv in v.time_stamps.items():
					for kkk, vvv in vv.items():
						new_dict[k].time_stamps[kk][kkk] = vvv - self.sched_config['time_ref']

			self.data = new_dict

	def num_iterations(self):
		return max([k[0] for k in self.data.keys()]) + 1

	def get_trajectory_data(self, increase=False, all_budgets=True, budget_bigger_is_better=False,
	                        non_decreasing_budget_limit=False):
		"""
		Get trajectory data.

		Parameters
		----------
		increase
			True or False, if True, the data is like score, otherwise the data is loss (descending)
		all_budgets
			whether we consider all budgets when getting trajectory data instead of considering only max_budget
		budget_bigger_is_better
			whether the bigger budget is better than smaller budget
		non_decreasing_budget_limit
			whether we can tolerate descending budget when getting trajectory data
		Returns
		-------

		"""
		trajectory = self.get_incumbent_trajectory(all_budgets=all_budgets,
		                                           budget_bigger_is_better=budget_bigger_is_better,
		                                           non_decreasing_budget_limit=non_decreasing_budget_limit)
		# print("Trajectory: {}".format(trajectory))
		assert 'times_finished' in trajectory, "Trajectory does not contains 'times_finished' key!"
		assert 'losses' in trajectory, "Trajectory does not contains 'losses' key!"
		times_finished = trajectory['times_finished']
		losses = trajectory['losses']
		config_ids = trajectory['config_ids']
		if increase:
			losses = list(map(lambda x: 1 - x, losses))
		return times_finished, losses, config_ids

	def plot_trajectory(self, increase=False, all_budgets=True, budget_bigger_is_better=False,
	                    non_decreasing_budget_limit=False):
		"""
		Plot trajectory using matplotlib.

		Parameters
		----------
		increase
		all_budgets
		budget_bigger_is_better
		non_decreasing_budget_limit

		Returns
		-------

		"""
		times_finished, losses, _ = self.get_trajectory_data(increase=increase, all_budgets=all_budgets,
		                                                     budget_bigger_is_better=budget_bigger_is_better,
		                                                     non_decreasing_budget_limit=non_decreasing_budget_limit)
		_plot(times_finished, losses)

	@staticmethod
	def _save_trajectory(directory, times_finished, losses):
		"""

		Parameters
		----------
		directory
		times_finished
		losses

		Returns
		-------

		"""
		traj_fn = os.path.join(directory, 'traj_{}_val.csv'.format(time.time()))
		with open(traj_fn, 'w') as fh:
			for time_f, loss in zip(times_finished, losses):
				fh.write("{}    {}\n".format(time_f, loss))

	def _hypTuner_model_fit(self, model, X=None, y=None):
		"""
		Thread method to perform fit on trajectory process.

		Parameters
		----------
		model
		X
		y

		Returns
		-------

		"""
		model.fit(X, y)
		self.fit_model = model

	def save_trajectory(self, directory, include_test=False, estimator=None,
	                    X_train=None, y_train=None, X_test=None, y_test=None, per_run_timebudget=360,
	                    all_budgets=True,
	                    budget_bigger_is_better=False,
	                    non_decreasing_budget_limit=False):
		"""
		Save trajectory into a file.

		Parameters
		----------
		directory
		include_test
		estimator
		X_train
		y_train
		X_test
		y_test
		per_run_timebudget
		all_budgets
		budget_bigger_is_better
		non_decreasing_budget_limit

		Returns
		-------

		"""
		times_finished, losses, config_ids = self.get_trajectory_data(increase=False,
		                                                              all_budgets=all_budgets,
		                                                              budget_bigger_is_better=budget_bigger_is_better,
		                                                              non_decreasing_budget_limit=non_decreasing_budget_limit)
		if include_test:
			assert estimator is not None
			assert X_train is not None
			assert X_test is not None
			traj_fn = os.path.join(directory, 'traj_{}_val_test.csv'.format(time.time()))
			with open(traj_fn, 'w') as fh:
				for time_f, loss, cfg_id in zip(times_finished, losses, config_ids):
					cfg = self.get_runs_by_id(cfg_id)[0].config
					model = estimator.new_estimator(cfg)
					try:
						self.fit_model = None
						t = threading.Thread(target=self._hypTuner_model_fit,
						                     args=(model, X_train, y_train,))
						t.setDaemon(True)
						t.start()
						t.join(per_run_timebudget)
						# model.fit(X_train, y_train)
						if self.fit_model is not None:
							y_hat = model.predict(X_test)
							test_acc = accuracy_score(y_test, y_hat)
						else:
							test_acc = 0.0
							print("fit超时")
					except Exception as e:
						print(e)
						test_acc = 0.0
					print("{} -> loss={:.4f}, val_score={:.4f}, test_score={:.4f}".format(cfg_id,
					                                                                      loss,
					                                                                      loss_to_score(
						                                                                      rule=self.sched_config['evaluation_rule'],
						                                                                      loss=loss),
					                                                                      test_acc))
					fh.write("{}    {}  {}\n".format(time_f, loss, test_acc))
		else:
			self._save_trajectory(directory=directory, times_finished=times_finished, losses=losses)

	@staticmethod
	def load_and_plot_trajectory_csv(directory, increase=False, filename=None):
		"""

		Parameters
		----------
		directory
		increase
		filename

		Returns
		-------

		"""
		if filename is None:
			lis = os.listdir(directory)
			trajs = list(filter(lambda x: "traj" in x and ".csv" in x, lis))
			if len(trajs) < 1:
				raise FileNotFoundError("No traj_*.csv file in {}".format(directory))
			sorted_trajs = sorted(trajs, reverse=True)
			traj_fn = os.path.join(directory, sorted_trajs[0])
		else:
			traj_fn = os.path.join(directory, filename)
		times_finished, losses = [], []
		with open(traj_fn, 'r') as fh:
			for line in fh.readlines():
				time_f, loss = line.strip().split()
				times_finished.append(float(time_f))
				losses.append(float(loss))
		# losses is descended stored
		if increase:
			losses = list(map(lambda x: 1 - x, losses))
		_plot(times_finished, losses)

	def get_incumbent_val_score(self):
		last_run = None
		try:
			last_run = self.get_runs_by_id(self.get_incumbent_id())[-1]
		except Exception as e:
			print(e)
		if last_run is None:
			return 0.0
		for info_key in last_run.info.keys():
			if 'val' in info_key:
				return last_run.info[info_key]
		raise RuntimeError("[Result] Should not arrive here! last_run.info contains no key like 'val_*'!")

	def summary(self, all_budgets=True):
		id2config = self.get_id2config_mapping()
		incumbent_id = self.get_incumbent_id(all_budgets=all_budgets)
		print("---------------- S U M M A Y ----------------")
		print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
		print('A total of %i runs where executed.' % len(self.get_all_runs(only_largest_budget=all_budgets)))
		print('The Best Configuration: {} -> {}'.format(incumbent_id, self.get_incumbent_config(all_budgets=all_budgets)))
		print(self.get_runs_by_id(incumbent_id)[-1])  # the last run of best config
		print("---------------------------------------------")


def checkFileExists(filename, overwrite):
	try:
		with open(filename, 'x'):
			pass
	except FileExistsError:
		if overwrite:
			with open(filename, 'w'):
				pass
		else:
			raise FileExistsError('The file %s already exists.' % filename)
	except Exception as e:
		raise e


def _plot(times_finished, losses):
	plt.plot(times_finished, losses)
	plt.show()


class JsonResultLogger(object):
	def __init__(self, directory, overwrite=False):
		"""
		Convenience logger for 'semi-live-results'

		Logger that writes job results into two files (configs.json and results.json).
		Both files contain proper json objects in each line.

		This version opens and closes the files for each result.
		This might be very slow if individual runs are fast and the
		filesystem is rather slow (e.g. a NFS).

		Parameters
		----------
		directory: string
			the directory where the two files 'configs.json' and
			'results.json' are stored
		overwrite: bool
			In case the files already exist, this flag controls the
			behavior:
				* True:   The existing files will be overwritten. Potential risk of deleting previous results
				* False:  A FileExistsError is raised and the files are not modified.
		"""

		os.makedirs(directory, exist_ok=True)

		self.config_fn = os.path.join(directory, 'configs.json')
		self.results_fn = os.path.join(directory, 'results.json')

		checkFileExists(self.config_fn, overwrite=overwrite)
		checkFileExists(self.results_fn, overwrite=overwrite)

		self.config_ids = set()

	def new_config(self, config_id, config, config_info):
		config = correct_config(config)
		if config_id not in self.config_ids:
			self.config_ids.add(config_id)
			with open(self.config_fn, 'a') as fh:
				fh.write(json.dumps([config_id, config, config_info]))
				fh.write('\n')

	def __call__(self, job):
		if job.cfg_id not in self.config_ids:
			# should never happen! TODO: log warning here!
			self.config_ids.add(job.cfg_id)
			with open(self.config_fn, 'a') as fh:
				fh.write(json.dumps([job.cfg_id, job.config, {}]))
				fh.write('\n')
		with open(self.results_fn, 'a') as fh:
			fh.write(json.dumps([job.cfg_id, job.budget_t[0], job.timestamps, job.result, job.exception]))
			fh.write("\n")


def logged_results_to_HypTuner_result(directory):
	"""
	Function to import logged 'live-results' and return a Result object

	You can load live run results with this function and the returned
	Result object gives you access to the results the same way
	a finished run would.

	Parameters
	----------
	directory: str
		the directory containing the results.json and config.json files

	Returns
	-------
	das.HypTuner.result.Result: :object:
	"""
	data = {}
	time_ref = float('inf')
	budget_set = set()

	with open(os.path.join(directory, 'configs.json')) as fh:
		for line in fh:
			line = json.loads(line)
			if len(line) == 3:
				config_id, config, config_info = line
			elif len(line) == 2:
				config_id, config, = line
				config_info = 'N/A'
			else:
				continue
			data[tuple(config_id)] = Datum(config=config, config_info=config_info)

	with open(os.path.join(directory, 'results.json')) as fh:
		for line in fh:
			config_id, budget, time_stamps, result, exception = json.loads(line)

			cfg_id = tuple(config_id)

			data[cfg_id].time_stamps[budget] = time_stamps
			data[cfg_id].results[budget] = result
			data[cfg_id].exceptions[budget] = exception

			budget_set.add(budget)
			time_ref = min(time_ref, time_stamps['submitted'])

	# infer the HyperBand configuration from the data
	budget_list = sorted(list(budget_set))

	sched_config = {'eta': None if len(budget_list) < 2 else budget_list[1] / budget_list[0],
		            'min_budget': min(budget_set),
		            'max_budget': max(budget_set),
		            'budgets': budget_list,
		            'max_SH_iter': len(budget_set),
		            'time_ref': time_ref}

	return Result([data], sched_config)


def correct_config(config):
	"""
	A config may contains key-value like ('SelectFpr__score_func': <function chi2 at 0x7fa3eace2ae8>), in this
	case, JSON can not dump the function object, which may cause a exception.
	We transform the non {float, int, string, bool, list, tuple, set, dict} value to string value.

	Parameters
	----------
	config

	Returns
	-------

	"""
	new_config = {}
	for cfg in config.keys():
		if not is_legal_config_item(config[cfg]):
			new_config[cfg] = str(config[cfg])
		else:
			new_config[cfg] = config[cfg]
	return new_config
