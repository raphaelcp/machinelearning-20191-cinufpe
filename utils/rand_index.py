import numpy as np 
from utils.convert_crisp import convert_crisp


def true_labels(n, K):
	tl = np.zeros(n, dtype=int)
	qtd_by_class = int(n/K)
	for i in range(n):
		# print(i, int(i/K))
		tl[i] = int(i/qtd_by_class)
	return tl

def new_crisp(crisp_matrix):
	n = len(crisp_matrix)
	crisp_vec = np.zeros(n, dtype=int)
	for i in range(n):
		crisp_vec[i] = np.argmax(crisp_matrix[i])
	return crisp_vec


if __name__ == '__main__':
	# print(true_labels(6, 3))

	crisp_matrix = np.array([
		[1,0,0], 
		[1,0,0],
		[1,0,0],
		[0,1,0],
		[0,0,1],
		[0,1,0],
	])

	print(rand_index(crisp_matrix))
