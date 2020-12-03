import ray
import psutil


def getCPUState(interval=1, fmt=True):
	if fmt:
		return " CPU: " + str(psutil.cpu_percent(interval)) + "%"
	return psutil.cpu_percent(interval)


def getMemoryState(fmt=True):
	phy_mem = psutil.virtual_memory()
	if fmt:
		line = "Memory: %5s%% %6s/%s" % (
			phy_mem.percent,
			str(int(phy_mem.used/1024/1024))+"M",
			str(int(phy_mem.total/1024/1024))+"M"
			)
		return line
	return phy_mem.percent


@ray.remote
def ray_fit_task(estimator, X, y, fit_params):
	estimator.fit(X, y, **fit_params)


if __name__ == '__main__':
	# while True:
	# 	print(getCPUState(1))
	print(getCPUState(interval=1, fmt=False))
	print(getMemoryState(fmt=False))


