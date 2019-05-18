import numpy as np

def weighted_dist(lmbd, ex_i, ex_g, D_matrices):
	"""
	sum de j=1 a p lambda[k][j]*d[j](e[i], g[k][j])

	lmbd -> matriz lambda
	ex_i -> exemplo i para calc das distancias
	ex_g -> vetor com a referencia para os representantes
	D_matrices -> matriz das distancias
	"""
	wsum = 0
	for j in range(len(ex_g)):
		wsum += lmbd[j] * D_matrices[j][ex_i][ex_g[j]]
	return wsum

def fuzzy_unit(i, k, W, G, D_matrices, K, m):
	"""
	"""
	fsum = 0
	for h in range(K):
		a = weighted_dist(W[k], i, G[k], D_matrices)
		b = weighted_dist(W[h], i, G[h], D_matrices)
		fsum += (a/b)**(1/(m-1))

	return 1/fsum

def fuzzy_matrix(W, G, D_matrices, K, m):
	"""
	calculo da eq 6
	"""
	n = len(D_matrices[0])
	U = np.zeros((n, K))

	for i in range(n):
		for k in range(K):
			U[i][k] = fuzzy_unit(i, k, W, G, D_matrices, K, m)

	return U

if __name__ == "__main__":
	i = 1
	k = 0
	W = [
		[1., 1., 1.],
		[1., 1., 1.],
	]
	G = [
		[0, 1, 1], # classe 0 nas views 1, 2 e 3
		[2, 2, 0],
		# [0, 2],
		# [1, 2],
		# [3, 1],
	]
	D_matrices =  [ # p*n*n
		[[1., 2., 3.],[4., 5., 6.],[7., 8., 9.]],
		[[1., 2., 3.],[4., 5., 6.],[7., 8., 9.]],
		[[1., 2., 3.],[4., 5., 6.],[7., 8., 9.]],
	]
	p = 3

	# print(weighted_dist(W[k], i, G[k], D_matrices))
	# print(fuzzy_partition(1, 0, W, G, D_matrices, 2, 1.6)+
	# 	fuzzy_partition(1, 1, W, G, D_matrices, 2, 1.6))
	print(fuzzy_matrix(W, G, D_matrices, 2, 1.6))