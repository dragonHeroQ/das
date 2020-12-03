from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
from das.ArchitectureSearch.Evaluator.Evaluator import DeepArchiEvaluator
from das.ArchitectureSearch.Optimizer.BayesianOptimizer import BayesianOptimizer
from das.ArchitectureSearch.SearchFramework.hyperband.HyperBandIteration import UCBIteration
learning_tool = DeepArchiLearningTool(n_block=2, n_classes=10, evaluation_rule='accuracy_score')
evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule='accuracy_score')
optimizer = BayesianOptimizer(parameter_space=learning_tool.get_parameter_space())

num_bandits = 10
all_bandits = dict([(i,
                     learning_tool.create_learning_tool(**optimizer.get_next_config()))
                    for i in range(10)])

# debug at 01/22 15:37
# all_bandits = {0: learning_tool.create_learning_tool(**{'b1_algo': 'QuadraticDiscriminantAnalysis', 'b1_num': 2,
#                                                         'b2_algo': 'RandomForestClassifier', 'b2_num': 4})}
# print(all_bandits)
# for bandit in all_bandits.values():
# 	print(bandit.get_configuration_space().get_space_names())

ucb_iter = UCBIteration(iteration_budget=7200, budget_type='time',
                        all_bandits=all_bandits, evaluator=evaluator,
                        sampling_strategy=None, stage_time_controller=None, max_num_stages=3,
                        evaluation_rule='accuracy_score',
                        per_run_timelimit=240.0, random_state=0)
# logger.setLevel('DEBUG')
from benchmarks.data.digits.load_digits import load_digits

x_train, x_test, y_train, y_test = load_digits()
print(x_train.shape)
ucb_iter.fit(x_train, y_train, debug=True)
# learning_curve = evaluator.load_learning_curve()
# print(learning_curve)
evaluator.plot_learning_curve()
# evaluator.save_learning_curve()
train_score, test_score = ucb_iter.refit_and_score(x_train, y_train, x_test, y_test)
print("Final Train Score={}, Test Score={}".format(train_score, test_score))

