import numpy as np

def arg_min(D_matrices, j, tmp_Uem, k):
	"""
	eq 4
	"""
	return (tmp_Uem[:,k] * D_matrices[j,:,:]).sum(axis=1).argmin()

def medioid_vector_selection(D_matrices, Uem, m):
	"""
	step 1 - algoritmo
	"""
	p = D_matrices.shape[0]
	K = Uem.shape[1]
	new_G = np.zeros((K, p), dtype=int)

	for k in range(K):
		for j in range(p):
			new_G[k,j] = arg_min(D_matrices, j, Uem, k)

	return new_G


if __name__ == '__main__':
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

	k = 1
	m = 1.6

	# print(D_matrices[0])
	# print(arg_min(D_matrices[0], U, k, m))
	print(medioid_vector_selection(D_matrices, U, m))
