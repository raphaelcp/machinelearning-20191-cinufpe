import numpy as np

def arg_min(diss_matrix, U, k, m):
	n = len(diss_matrix)
	amin = np.inf

	for h in range(n):
		dsum = 0
		for i in range(n):
			dsum += U[i][k]**m * diss_matrix[i][h]

		if amin > dsum:
			amin = dsum

	return amin

# def medioid_vector_selection(diss_matrices):
		


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

	print(diss_matrices)
	print(arg_min(diss_matrices[0], U, k, m))
