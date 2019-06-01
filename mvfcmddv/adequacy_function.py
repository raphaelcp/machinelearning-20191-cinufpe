import numpy as np
from .fuzzy_partition import weighted_dist


# v= (G,W,U)
def adequacy(v, m, D_matrices, Uem):
	"""
	Função calcula resultado de função objetivo, a ser minimizada. Onde:
		v[0] = G, matriz de protótipos
		v[1] = W, matriz de pesos
		v[2] = U, matriz de grau de pertencimento
		m = 1.6,
		D_matrices = matrizes de dissimilaridade
	"""
	K = len(v[1])
	n = len(v[2])
	asum = 0;

	for k in range(K):
		for i in range(n):
			asum += Uem[i,k] * weighted_dist(v[1], i, v[0], D_matrices, k)

	return asum

if __name__ == '__main__':
	np.random.seed(42)

	D_matrices = np.random.rand(3,9,9)


	G = [[7, 8, 3],[4, 5, 6]]

	W = [
		[1., 1., 1.,],
		[1., 1., 1.,]
	]
	U = [
		[0.76046769, 0.23953231],
		[0.86188325, 0.13811675],
		[0.23953231, 0.76046769]
	]
	v=[G,W,U]

	print(adequacy(v, 1.6, D_matrices))
