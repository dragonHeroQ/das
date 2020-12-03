import hashlib


def get_hash(o: object):
	if isinstance(o, str):
		md5 = hashlib.md5()
		md5.update(o.encode())
		md5_str = md5.hexdigest()
		hash_value = 1
		for ch in md5_str:
			hash_value *= ord(ch)
		return hash_value % 1000000007
	elif isinstance(o, int):
		return o % 1000000007
	elif isinstance(o, dict):
		items_list = o.items()
		items_list = sorted(items_list, key=lambda x: x[0])
		hash_string = ""
		for k, v in items_list:
			hash_string += "{}:{}".format(k, v)
		return get_hash(hash_string)
	else:
		return hash(o) % 1000000007


if __name__ == '__main__':
	print(get_hash("2/3-4"))
