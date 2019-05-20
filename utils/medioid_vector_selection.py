import numpy as np

def arg_min(D, U, k, m):
	"""
	eq 4
	"""
	# print('init arg_min')
	n = len(D)
	amin = np.inf
	ind_l = -1

	tmp_Uem = np.zeros(n)
	for i in range(n):
		tmp_Uem[i] = U[i][k]**m

	for h in range(n):
		dsum = 0
		for i in range(n):
			dsum += tmp_Uem[i] * D[i][h]

		# print('%f > %f'%(amin, dsum))
		if amin > dsum:
			amin = dsum
			ind_l = h
			# print('%d - %f'%(ind_l, amin))

	# print('end arg_min')
	return ind_l

def medioid_vector_selection(D_matrices, U, m):
	"""
	step 1 - algoritmo
	"""
	p = len(D_matrices)
	K = len(U[0])
	new_G = np.zeros((K, p), dtype=int)

	for k in range(K):
		for j in range(p):
			new_G[k][j] = arg_min(D_matrices[j], U, k, m)

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
