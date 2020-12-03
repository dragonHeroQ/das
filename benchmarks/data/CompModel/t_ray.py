import ray

ray.init()

for i in range(0, 10):
	lisa_id = ray.put("lisa")


