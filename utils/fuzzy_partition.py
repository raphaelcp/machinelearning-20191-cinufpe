import numpy as np

def weighted_dist(W, i, G, D_matrices, kh):
	"""
	sum de j=1 a p lambda[k][j]*d[j](e[i], g[k][j])

	lmbd -> matriz lambda
	i -> exemplo i para calc das distancias
	ex_g -> vetor com a referencia para os representantes
	D_matrices -> matriz das distancias
	"""
	wsum = 0
	p = D_matrices.shape[0]
	for j in range(p):
		wsum += W[kh,j] * D_matrices[j,i,G[kh,j]]
	return wsum

def fuzzy_unit(i, k, W, G, D_matrices, K, m, eps):
	"""
	"""
	fsum = 0

	a = weighted_dist(W, i, G, D_matrices, k)
	for h in range(K):
		b = weighted_dist(W, i, G, D_matrices, h)
		if b < eps:
			return 0
		fsum += (a/b)**(1./(m-1))

	return 1./fsum

def fuzzy_matrix(W, G, D_matrices, K, m, old_U, eps=10**(-10)):
	"""
	calculo da eq 6
	"""
	n = D_matrices.shape[1]
	new_U = np.zeros((n, K))

	for i in range(n):
		for k in range(K):
			tmp = fuzzy_unit(i, k, W, G, D_matrices, K, m, eps)
			if tmp == 0:
				return old_U
			new_U[i,k] = tmp

	return new_U

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
	print(fuzzy_matrix(W, G, D_matrices, 2, 1.6), [])
