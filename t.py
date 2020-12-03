import sys
import multiprocessing
import attr
import socket
import os.path as osp
import numpy as np
import random
from das.BaseAlgorithm.Classification.ExtraTreesClassifier import ExtraTreesClassifier
from das.BaseAlgorithm.Classification.RandomForestClassifier import RandomForestClassifier
from benchmarks.data.yeast.load_yeast import load_yeast


import ray
#
#
@ray.remote
def f(a):
	print(a)
#
#
# # ray.init()
#
a = np.array([[1, 2],
              [3, 2]])
#
# # ray.put(a)
#
# # ray.put(a)
#
#
def put_process():
	ray.init(redis_address="192.168.100.35:6379")
	ID1 = ray.put(a)
	print(ID1)


p = multiprocessing.Process(target=put_process)

p.start()
p.join()

p2 = multiprocessing.Process(target=put_process)

p2.start()
p2.join()

# # 0000000073bc2a9302e31495c948190ea9810f60
# # 00000000393f26cb712f101240388a2b773a0f0e
# # 0000000068338be4f879b0b9e93dfeea817d09ed
# ray.init(redis_address="192.168.100.35:6379")
#
# ha = f.remote(a)
# ha_ = ray.get(ha)
#
# ga = f.remote(a)
# ga_ = ray.get(ga)
# print(ha_, ga_)
#
# ray.shutdown()
#
# ray.init(redis_address="192.168.100.35:6379")
#
# ha = f.remote(a)
# hat = ray.get(ha)
#
# ga = f.remote(a)
# gat = ray.get(ga)
#
# print(hat, gat)
