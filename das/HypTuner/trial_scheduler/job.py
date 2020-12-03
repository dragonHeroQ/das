import time


class Job(object):
	"""
	Job is a description of a trial.
	"""
	def __init__(self,
	             cfg_id,
	             timeout=None,
	             config=None,
	             budget_type="time",
	             budget_t=None,
	             working_directory=".",
	             estimator=None,
	             X=None,
	             y=None,
	             evaluation_rule=None,
	             validation_strategy=None,
	             validation_strategy_args=None,
	             worst_score=None,
	             task='classification',
	             **kwargs):

		# running meta data
		self.cfg_id = cfg_id
		self.timeout = timeout
		self.config = config
		self.budget_type = budget_type
		self.budget_t = budget_t
		self.working_directory = working_directory

		# running stuff
		self.estimator = estimator
		self.X = X
		self.y = y
		self.evaluation_rule = evaluation_rule   # for clustering
		self.validation_strategy = validation_strategy
		self.validation_strategy_args = validation_strategy_args
		self.worst_score = worst_score
		self.task = task

		# other stuff
		self.kwargs = kwargs
		self.timestamps = {}
		self.result = None
		self.exception = None
		self.worker_name = None

	def time_it(self, which_time):
		self.timestamps[which_time] = time.time()

	def __repr__(self):
		return (
			"job_id (cfg_id): " + str(self.cfg_id) + "\n" +
			"kwargs: " + str(self.kwargs) + "\n" +
			"evaluation_rule: " + str(self.evaluation_rule) + "\n" +
			"result: " + str(self.result) + "\n" +
			"exception: " + str(self.exception) + "\n" +
			"timestamps: " + str(self.timestamps)
		)
