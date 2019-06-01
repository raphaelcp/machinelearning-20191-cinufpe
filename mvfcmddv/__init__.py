import numpy as np
from .initialization import *
from .fuzzy_partition import fuzzy_matrix
from .adequacy_function import adequacy
from .medioid_vector_selection import medioid_vector_selection
from .vectors_relevance_weights import vectors_relevance_weights
import time

def MVFCMddV(D_matrices, K, m=1.6, eps=10**(-10), T=150):
	t = 0
	p = D_matrices.shape[0]
	n = D_matrices.shape[1]

	#Inicialização dos dados
	G = init_gmedoid(K, p, n)
	W = init_weights(K, p)
	U = fuzzy_matrix(W, G, D_matrices, K, m, [])
	# print(U)
	# exit()

	u = [];
	u.append(adequacy([G, W, U], m, D_matrices, U**m))
	# print(u)
	while True:
		t += 1
		print('time %d'%(t))

		microtime = time.time()
		Uem = U**m
		# print('- calc Uem %.3f'%(time.time()-microtime))

		# print(G)
		microtime = time.time()
		G = medioid_vector_selection(D_matrices, Uem, m)
		# np.savetxt('output/G'+str(t), G, fmt='%i')
		# print('- calc G %.3f'%(time.time()-microtime))
		# u.append(adequacy([G, W, U], m, D_matrices, Uem))

		microtime = time.time()
		W = vectors_relevance_weights(D_matrices, G, Uem, m)
		# np.savetxt('output/W'+str(t), W, fmt='%.7f')
		# print('- calc W %.3f'%(time.time()-microtime))
		# u.append(adequacy([G, W, U], m, D_matrices, Uem))

		microtime = time.time()
		U = fuzzy_matrix(W, G, D_matrices, K, m, U)
		# np.savetxt('output/U'+str(t), U, fmt='%.7f')
		# print('- calc U %.3f'%(time.time()-microtime))
		# u.append(adequacy([G, W, U], m, D_matrices, Uem))

		microtime = time.time()
		u.append(adequacy([G, W, U], m, D_matrices, Uem))
		# print('- calc u %.3f'%(time.time()-microtime))
		# print(u)
		if abs(u[t]-u[t-1]) < eps or t >= T:
			break

	return {
		'performance': u[t],
		'G': G,
		'W': W,
		'U': U,
	}
	# crisp = convert_crisp(U)
	# np.savetxt('output/crisp', crisp, fmt='%d')

	# qtdpclass = len(crisp)/K
	# resume_matrix = np.zeros((K, K))
	# for i in range(K):
	# 	subset = crisp[int(i*qtdpclass):int((i+1)*qtdpclass)]
	# 	resume_matrix[i] = np.sum(subset, axis=0)

	# np.savetxt('output/resume', resume_matrix, fmt='%d')