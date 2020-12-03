import copy
from das.HypTuner.trial_scheduler.job import Job
from das.HypTuner.iteration.base_iteration import BaseIteration


class WarmStartIteration(BaseIteration):
	"""
	Iteration that imports a previous Result for warm starting
	"""

	def __init__(self, result, config_generator):

		self.is_finished = False
		self.stage = 0

		id2conf = result.get_id2config_mapping()
		delta_t = - max(map(lambda r: r.time_stamps['finished'], result.get_all_runs()))

		super().__init__(-1, [len(id2conf)], [None], None)

		for i, cfg_id in enumerate(id2conf):

			new_id = self.add_configuration(config=copy.deepcopy(id2conf[cfg_id]['config']),
			                                config_info=id2conf[cfg_id]['config_info'])

			for r in result.get_runs_by_id(cfg_id):

				j = Job(new_id, config=id2conf[cfg_id]['config'], budget_t=(r.budget, 0, 1))

				j.result = {'loss': r.loss, 'info': r.info}
				j.error_logs = r.error_logs

				for k, v in r.time_stamps.items():
					j.timestamps[k] = v + delta_t

				self.register_result(j, skip_sanity_checks=True)

				config_generator.new_result(j, update_model=(i == len(id2conf)-1))

		# mark as finished, as no more runs should be executed from these runs
		self.is_finished = True

	def fix_timestamps(self, time_ref):
		"""
		Manipulates internal time stamps such that the last run ends at time 0
		"""

		for k, v in self.data.items():
			for kk, vv in v.time_stamps.items():
				for kkk, vvv in vv.items():
					self.data[k].time_stamps[kk][kkk] += time_ref

	def _advance_to_next_stage(self, config_ids, losses):
		pass
