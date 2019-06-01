import numpy as np
from .initialization import *
from .fuzzy_partition import fuzzy_matrix
from .adequacy_function import adequacy
from .medioid_vector_selection import medioid_vector_selection
from .vectors_relevance_weights import vectors_relevance_weights
import time

def MVFCMddV(D_matrices, K, m=1.6, eps=10**(-10), T=150):
	microtime = time.time()
	t = 0
	p = D_matrices.shape[0]
	n = D_matrices.shape[1]

	# Inicialização dos dados
	G = init_gmedoid(K, p, n)
	W = init_weights(K, p)
	U = fuzzy_matrix(W, G, D_matrices, K, m, [])

	J = [];
	J.append(adequacy([G, W, U], m, D_matrices, U**m))
	print("J[%d]=%.9f - %.3fs"%(t, J[t], time.time()-microtime))

	while True:
		t += 1
		microtime = time.time()

		Uem = U**m

		G = medioid_vector_selection(D_matrices, Uem, m)
		W = vectors_relevance_weights(D_matrices, G, Uem, m)
		U = fuzzy_matrix(W, G, D_matrices, K, m, U)

		J.append(adequacy([G, W, U], m, D_matrices, Uem))

		print("J[%d]=%.9f - %.3fs"%(t, J[t], time.time()-microtime))
		if abs(J[t]-J[t-1]) < eps or t >= T:
			break

	return {
		'performance': J[t],
		'J': J,
		'G': G,
		'W': W,
		'U': U,
	}
