import pickle
import numpy as np

for data in ['superconduct']:

	mse_dict = pickle.load(open('{}_mses.pkl'.format(data), 'rb'))

	# print(mse_dict)

	sorted_mses = sorted(mse_dict.items(), key=lambda x: x[1], reverse=False)
	print("First 10 winners:")
	for k, v in sorted_mses[:10]:
		print(k, v)

	filtered_mses = list(filter(lambda x: x[1][0] != np.inf, mse_dict.items()))
	timecosts = list(map(lambda x: x[1][1], filtered_mses))
	sorted_mses = sorted(filtered_mses, key=lambda x: x[1][1], reverse=True)
	# print("First 10 TimeCost Winners:")
	# for k, v in sorted_mses[:10]:
	# 	print(k, v)

	print("================= {} ===================".format(data))
	print("Max Valid TimeCost: {}".format(sorted_mses[0][1][1]))
	print("Valid TimeCost MEAN = {}, STD = {}".format(np.mean(timecosts), np.std(timecosts)))
	print("MEAN + STD*15 = {}".format([np.mean(timecosts) + np.std(timecosts) * i for i in range(15, 0, -1)]))
