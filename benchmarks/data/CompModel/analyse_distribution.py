import math
import pickle
import random
import numpy as np


def grubus(vals):
	# 100
	# 90% 95% 97.5% 99.0% 99.5%
	# 3.017 3.207 3.383 3.600 3.754
	critic_value = 3.6 + (len(vals)-100)*0.003
	print("critic_value: {}".format(critic_value))
	new_vals = []
	mean_value = float(np.mean(vals))
	std_value = float(np.std(vals))
	print("Mean = {}, STD = {}".format(mean_value, std_value))
	for v in vals:
		if (v-mean_value)/std_value > critic_value:
			continue
		else:
			new_vals.append(v)
	return new_vals


def minmax(vals):
	max_val = np.max(vals)
	min_val = np.min(vals)
	delta = max_val-min_val

	probs = list(map(lambda x: float(max_val-x)/delta, vals))
	print("Probs: ", probs)
	vals_mask = [1 if random.random() < probs[i] and vals[i] < min_val*1000 else 0 for i in range(len(vals))]
	print("vals_mask: ", vals_mask)
	new_vals = []
	for i, val in enumerate(vals):
		if vals_mask[i]:
			new_vals.append(val)
	print(new_vals)
	ex = [math.exp(-x) for x in new_vals]
	sum_ex = sum(ex)
	resources = list(map(lambda x: math.exp(-x)/sum_ex, new_vals))
	print("resources: ", resources)
	Total_Resources = 720
	Time_Relocated = list(map(lambda x: x*Total_Resources, resources))
	# Time_Relocated = list(filter(lambda x: x > 0.0, Time_Relocated))
	print(len(Time_Relocated), "Time Relocated: ", Time_Relocated)
	return new_vals


data = 'mg'

mse_dict = pickle.load(open('../CompModel_PKLS/{}_mses.pkl'.format(data), 'rb'))

# print(mse_dict)

sorted_mses = sorted(mse_dict.items(), key=lambda x: x[1][0], reverse=False)
# print("First 10 winners:")

vals = []
for k, v in sorted_mses:
	if v[0] != np.inf:
		vals.append(v[0])
	else:
		vals.append(vals[-1])

print(vals)

print("Before MinMax: {}".format(len(vals)))
vals = minmax(vals)
print("After MinMax: {}".format(len(vals)))

import matplotlib.pyplot as plt
plt.scatter(x=np.array(np.arange(len(vals))), y=vals)

plt.xlabel("x")
plt.ylabel("y")
plt.show()
