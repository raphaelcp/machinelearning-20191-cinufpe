import numpy as np
from utils.normalized_dissimilarity import normalized_dissimilarity
from utils.initialization import *
from utils.fuzzy_partition import fuzzy_matrix
from utils.adequacy_function import adequacy
from utils.medioid_vector_selection import medioid_vector_selection
from utils.vectors_relevance_weights import vectors_relevance_weights
from utils.vectors_relevance_weights import vectors_relevance_weights

def MVFCMddV(D_matrices, K=10, m=1.6, eps=10**(-10), T=150):
	t = 0
	p = len(D_matrices)
	n = len(D_matrices[0])
	G = init_gmedoid(K, p, n)
	W = init_weights(K, p)
	U = fuzzy_matrix(W, G, D_matrices, K, m)
	# print(U)
	# exit()

	u = [];
	u.append(adequacy([G, W, U], m, D_matrices))
	# print(u)
	while True:
		t += 1
		print('time %d'%(t))

		G = medioid_vector_selection(D_matrices, U, m)

		W = vectors_relevance_weights(D_matrices, G, U, m)

		U = fuzzy_matrix(W, G, D_matrices, K, m)

		u.append(adequacy([G, W, U], m, D_matrices))

		if abs(u[t]-u[t-1]) < eps or t >= T:
			break


def main():
	D_matrices = normalized_dissimilarity(
		list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')
	)
	MVFCMddV(D_matrices)


if __name__ == '__main__':
	np.random.seed(42)
	main()
