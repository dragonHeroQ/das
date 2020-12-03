from das.performance_evaluation import score_to_loss


class Datum(object):
	"""
	Iteration Datum.

	Parameters
	----------
	config
	config_info
	results
	time_stamps
	exceptions
	status
		five status, 'TERMINATED', 'REVIEW', 'CRASHED', 'QUEUED', 'COMPLETED'
	budget
	"""
	def __init__(self,
	             config,
	             config_info,
	             results=None,
	             time_stamps=None,
	             exceptions=None,
	             status='QUEUED',
	             budget=0):
		self.config = config
		self.config_info = config_info
		self.results = results if results is not None else {}
		self.time_stamps = time_stamps if time_stamps is not None else {}
		self.exceptions = exceptions if exceptions is not None else {}
		self.status = status
		self.budget = budget

	def __repr__(self):
		return (
			"\nconfig:{}\n".format(self.config) +
			"config_info:\n{}\n".format(self.config_info) +
			"losses/results:\n" +
			''.join(["{}: {}\n".format(k, v['loss']) for k, v in self.results.items()]) +
			"time stamps: {}\n".format(self.time_stamps) +
			"status: {}".format(self.status)
		)

	@staticmethod
	def build_iteration(config, val_score, budget, evaluation_rule="accuracy_score"):
		"""
		Build wart_start iteration from config->val_score.

		Parameters
		----------
		config
		val_score
		budget
		evaluation_rule

		Returns
		-------
			iteration datum instance
		"""
		config_info = {'model_based_pick': False}
		results = {budget: {'loss': score_to_loss(rule=evaluation_rule, score=val_score),
		                    'info': {'val_acc': val_score,
		                             'exception': None}}}
		timestamps = {budget: {'submitted': 0.0,
		                       'started': 0.0,
		                       'finished': 0.0}}
		return Datum(config, config_info=config_info,
		             results=results,
		             time_stamps=timestamps,
		             exceptions=None,
		             status='FINISHED',
		             budget=budget)


