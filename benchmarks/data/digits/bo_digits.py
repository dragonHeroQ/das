import sys
sys.path.append("../")
sys.path.append("../../")
from das.ArchitectureSearch.SearchFramework.smbo.BayesOptimization import BayesOptimization
from das.ArchitectureSearch.LearningTool.DeepArchiLearningTool import DeepArchiLearningTool
from das.ArchitectureSearch.Evaluator.Evaluator import DeepArchiEvaluator

learning_tool = DeepArchiLearningTool(n_block=2, n_classes=10, evaluation_rule='accuracy_score')
evaluator = DeepArchiEvaluator(n_folds=3, evaluation_rule='accuracy_score')
BO = BayesOptimization(evaluator=evaluator, learning_tool=learning_tool,
                       total_budget=7200, budget_type='time',
                       per_run_timelimit=240.0, evaluation_rule='accuracy_score',
                       random_state=0)
# logger.setLevel('DEBUG')
from benchmarks.data.digits.load_digits import load_digits

x_train, x_test, y_train, y_test = load_digits()
print(x_train.shape)
BO.fit(x_train, y_train)
# learning_curve = evaluator.load_learning_curve()
# print(learning_curve)
# evaluator.plot_learning_curve()
# evaluator.save_learning_curve()
train_score, test_score = BO.refit_and_score(x_train, y_train, x_test, y_test)
print("Final Train Score={}, Test Score={}".format(train_score, test_score))

