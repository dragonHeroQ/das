from das.ArchitectureSearch.SearchFramework.BaseSearchFramework import BaseSearchFramework


class HyperBand(BaseSearchFramework):
	def __init__(self,
	             optimizer=None,
	             optimizer_class=None,
	             optimizer_params=None,
	             evaluator=None,
	             learning_tool=None,
	             sampling_strategy=None,
	             search_space=None,
	             total_budget=10,
	             budget_type="trial",
	             per_run_timelimit=240.0,
	             evaluation_rule=None,
	             time_ref=None,
	             worst_loss=None,
	             random_state=None,
	             task=None,
	             n_classes=None,
	             **kwargs):
		if optimizer is not None:
			self.optimizer = optimizer
		else:
			assert learning_tool is not None, "You should set learning_tool for SMBO(BaseSearchFramework)"
			parameter_space = learning_tool.get_parameter_space()
			optimizer_params = {} if optimizer_params is None else optimizer_params
			self.optimizer = optimizer_class(parameter_space=parameter_space, **optimizer_params)
		super(HyperBand, self).__init__(optimizer_class=optimizer_class,
		                                optimizer_params=optimizer_params,
		                                evaluator=evaluator,
		                                learning_tool=learning_tool,
		                                sampling_strategy=sampling_strategy,
		                                search_space=search_space,
		                                total_budget=total_budget,
		                                budget_type=budget_type,
		                                per_run_timelimit=per_run_timelimit,
		                                evaluation_rule=evaluation_rule,
		                                time_ref=time_ref,
		                                worst_loss=worst_loss,
		                                task=task,
		                                n_classes=n_classes,
		                                random_state=random_state,
		                                **kwargs)
		self.n_classes = n_classes
		self.records = []
		self.best_config = None
		self.best_num_layers = None

	def fit(self, X, y, **fit_params):
		pass

	def refit(self, X, y, **refit_params):
		pass

	def refit_transform(self, X, y, X_test, **refit_params):
		pass

	def refit_and_score(self, X, y, X_test, y_test, **refit_params):
		pass
