import numpy as np 

def crisp_obj(crisp, K):
	obj_list = [[] for i in range(K)]
	for i in range(len(crisp)):
		obj_list[crisp[i]].append(i)

	return obj_list

if __name__ == '__main__':
	





