from das.ArchitectureSearch.Evaluator.BaseEvaluator import BaseEvaluator


learning_curve = BaseEvaluator.load_learning_curve(
	"WHATBEG_bo_airfoil_r0_time_200_TL_240_02191913(0.9409710734313985).lcv", base_dir='./lcvs')

print(learning_curve)

sorted_learning_curve = sorted(learning_curve.items(), key=lambda x: x[0])

print(sorted_learning_curve[0])

learning_tool = sorted_learning_curve[0][1]['learning_tool']

to_evaluate_learning_tool = learning_tool.create_learning_tool(**learning_tool.hyper_params)




