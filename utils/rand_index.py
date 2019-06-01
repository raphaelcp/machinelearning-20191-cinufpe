import numpy as np
from .convert_crisp import convert_crisp
from scipy.misc import comb

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

def rand_index(crisp_matrix):
	clusters = true_labels(crisp_matrix.shape[0], crisp_matrix.shape[1])
	n = len(crisp_matrix)
	classes = np.zeros(n, dtype=int)
	for i in range(n):
		classes[i] = np.argmax(crisp_matrix[i])

	tp_plus_fp = comb(np.bincount(clusters), 2).sum()
	tp_plus_fn = comb(np.bincount(classes), 2).sum()
	A = np.c_[(clusters, classes)]
	tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
		for i in set(clusters))
	fp = tp_plus_fp - tp
	fn = tp_plus_fn - tp
	tn = comb(len(A), 2) - tp - fp - fn

	return (tp + tn) / (tp + fp + fn + tn)

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
