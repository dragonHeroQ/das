import das
import logging
from das.ArchitectureSearch.SearchFramework.smbo.SMBO import SMBO
from das.ArchitectureSearch.Optimizer.BayesianOptimizer import BayesianOptimizer

logger = logging.getLogger(das.logger_name)


class BayesOptimization(SMBO):

	def __init__(self,
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
	             **kwargs
	             ):
		super(BayesOptimization, self).__init__(optimizer_class=BayesianOptimizer,
		                                        optimizer_params=None,
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


if __name__ == '__main__':
	from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
	from das.ArchitectureSearch.Evaluator.DeepArchiEvaluator import DeepArchiEvaluator

	learning_tool = DeepArchiLearningTool(n_block=2, evaluation_rule='accuracy_score')
	evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule='accuracy_score')
	BO = BayesOptimization(evaluator=evaluator, learning_tool=learning_tool,
	                       total_budget=2, budget_type='trial',
	                       per_run_timelimit=240.0, evaluation_rule='accuracy_score',
	                       random_state=0)
	# logger.setLevel('DEBUG')
	from benchmarks.data.digits.load_digits import load_digits

	x_train, x_test, y_train, y_test = load_digits()
	print(x_train.shape)
	BO.fit(x_train, y_train)
	# learning_curve = evaluator.load_learning_curve()
	# print(learning_curve)
	evaluator.plot_single_learning_curve()
	# evaluator.save_learning_curve()
	train_score, test_score = BO.refit_and_score(x_train, y_train, x_test, y_test)
	print("Final Train Score={}, Test Score={}".format(train_score, test_score))
