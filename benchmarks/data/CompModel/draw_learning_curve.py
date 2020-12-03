import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_name = 'mg'
rng = 0
pickle_name = "{}_mses_{}".format(data_name, rng)

learning_curve = pickle.load(open("{}.lcv".format(pickle_name), 'rb'))

sorted_items = sorted(learning_curve.items(), key=lambda x: x[0])

print(sorted_items)

times = list(map(lambda x: x[0], sorted_items))
val_losses = list(map(lambda x: x[1], sorted_items))
min_val_losses = np.inf
for i in range(len(val_losses)):
	if val_losses[i] < min_val_losses:
		min_val_losses = val_losses[i]
	val_losses[i] = min_val_losses

plt.yscale('log')
plt.ylim([1e-2, 1.0])
plt.xscale('log')
plt.plot(times, val_losses)
plt.show()
