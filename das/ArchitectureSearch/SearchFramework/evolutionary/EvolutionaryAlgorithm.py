import das
import time
import numpy as np
import logging
from das.util.proba_utils import from_probas_to_performance
from das.ArchitectureSearch.SearchFramework.BaseSearchFramework import BaseSearchFramework

logger = logging.getLogger(das.logger_name)


class EvolutionaryAlgorithm(BaseSearchFramework):

	def __init__(self,
	             optimizer=None,
	             optimizer_class=None,
	             optimizer_params=None,
	             evaluator=None,
	             learning_tool=None,
	             sampling_strategy=None,
	             search_space=None,
	             P=100,
	             S=25,
	             identity_proba=0.05,
	             total_budget=10,
	             budget_type="trial",
	             per_run_timelimit=240.0,
	             evaluation_rule=None,
	             cross_validator=None,
	             time_ref=None,
	             worst_loss=None,
	             task=None,
	             random_state=None,
	             **kwargs):
		if optimizer is not None:
			self.optimizer = optimizer
		else:
			assert learning_tool is not None, "You should set learning_tool for EvolutionaryAlgorithm(BaseSearchFramework)"
			parameter_space = learning_tool.get_parameter_space()
			optimizer_params = {} if optimizer_params is None else optimizer_params
			self.optimizer = optimizer_class(parameter_space=parameter_space, **optimizer_params)
		super(EvolutionaryAlgorithm, self).__init__(optimizer=self.optimizer,
		                                            evaluator=evaluator,
		                                            learning_tool=learning_tool,
		                                            sampling_strategy=sampling_strategy,
		                                            search_space=search_space,
		                                            total_budget=total_budget,
		                                            budget_type=budget_type,
		                                            per_run_timelimit=per_run_timelimit,
		                                            evaluation_rule=evaluation_rule,
		                                            cross_validator=cross_validator,
		                                            time_ref=time_ref,
		                                            worst_loss=worst_loss,
		                                            task=task,
		                                            random_state=random_state,
		                                            **kwargs)
		self.P = P  # indicates the size of initial population
		self.S = S  # indicates the number of competitors each round(cycle)
		self.identity_proba = identity_proba  # probability of mutating nothing

	def init_population(self, X, y):
		pass

	def tournament_selection(self):
		pass

	def eliminate_member(self):
		pass

	def fit(self, X, y, **fit_params):
		pass

	def refit(self, X, y, **refit_params):
		pass

	def refit_transform(self, X, y, X_test, **refit_params):
		pass

	def refit_and_score(self, X, y, X_test, y_test, **refit_params):
		pass
