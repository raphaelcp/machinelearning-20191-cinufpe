import numpy as np

def sum_uk(D_matrices, hj, tmp_Uem, k, g):
	"""
	"""
	n = D_matrices.shape[1]

	dsum = 0
	for i in range(n):
		dsum += tmp_Uem[i,k] * D_matrices[hj,i,g]

	return dsum

def relevance_weights(D_matrices, G, tmp_Uem, k, j):
	"""
	eq 5
	"""
	p = len(D_matrices)
	prod = 1
	for h in range(p):
		prod *= sum_uk(D_matrices, h, tmp_Uem, k, G[k,h])

	return (prod**(1./p))/sum_uk(D_matrices, j, tmp_Uem, k, G[k,j])

def vectors_relevance_weights(D_matrices, G, Uem, m):
	"""
	step 2 - algoritmo
	"""
	K = len(G)
	p = len(D_matrices)
	new_W = np.zeros((K, p))

	for k in range(K):
		for j in range(p):
			new_W[k,j] = relevance_weights(D_matrices, G, Uem, k, j)

	return new_W;


if __name__ == "__main__":
	np.random.seed(42)

	D_matrices = np.random.rand(3,9,9)
	U = [
		[0.76046769, 0.23953231],
		[0.86188325, 0.13811675],
		[0.23953231, 0.76046769],
		[0.76046769, 0.23953231],
		[0.86188325, 0.13811675],
		[0.23953231, 0.76046769],
		[0.76046769, 0.23953231],
		[0.86188325, 0.13811675],
		[0.23953231, 0.76046769],
	]

	m = 1.6

	W = [
		[1., 1., 1.],
		[1., 1., 1.],
	]
	G = [
		[0, 1, 1], # classe 0 nas views 1, 2 e 3
		[2, 2, 0],
	]

	p = 3


	print(sum_uk(D_matrices[0], U, 0, m, G[0][0]))
	print(relevance_weights(D_matrices, G, U, 0, 0, m))
	new_W = vectors_relevance_weights(D_matrices, G, U, m)
	print(new_W)

	print('\nverificacao')
	for k in range(len(new_W)):
		tmp = 1
		for j in range(len(new_W[k])):
			if (new_W[k][j] <= 0):
				print('elemento na posicao %d,%d nao eh maior que 0'%(k, j))
			tmp *= new_W[k][j]
		# print(new_W[k])
		print(tmp)
