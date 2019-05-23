import numpy as np 
from sklearn.metrics.cluster import adjusted_rand_score
from utils.convert_crisp import convert_crisp


def true_labels(n, K):
	tl = np.zeros((n))
	qtd_by_class = int(n/K)
	for i in range(n):
		# print(i, int(i/K))
		tl[i] = int(i/qtd_by_class)
	return tl


def adjusted_rand_index(crisp_matrix):
	n = len(crisp_matrix)
	new_crisp = np.zeros(n)
	for i in range(n):
		new_crisp[i] = np.argmax(crisp_matrix[i])
	return adjusted_rand_score(new_crisp, true_labels(crisp_matrix.shape[0], crisp_matrix.shape[1]))

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
