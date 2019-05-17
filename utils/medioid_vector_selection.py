import numpy as np

def arg_min(diss_matrix, U, k, m):
	"""
	eq 4
	"""
	n = len(diss_matrix)
	amin = np.inf
	ind_l = -1

	for h in range(n):
		dsum = 0
		for i in range(n):
			dsum += U[i][k]**m * diss_matrix[i][h]

		if amin > dsum:
			amin = dsum
			ind_l = h

	return ind_l

def medioid_vector_selection(diss_matrices, U, m):
	p = len(diss_matrices)
	K = len(U[0])
	new_G = np.zeros((K, p), dtype=int)

	for k in range(K):
		for j in range(p):
			new_G[k][j] = arg_min(diss_matrices[k], U, k, m)


	return new_G


if __name__ == '__main__':
	np.random.seed(42)

	diss_matrices = np.random.rand(3,9,9)
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

	# print(diss_matrices[0])
	# print(arg_min(diss_matrices[0], U, k, m))
	print(medioid_vector_selection(diss_matrices, U, m))
