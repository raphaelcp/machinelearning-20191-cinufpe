import numpy as np

def init_gmedoid(K, p, n):
	g_medoid_matrix = [[]]*p
	for i in range(p):
		g_medoid_vector = []
		while len(g_medoid_vector) < K:
			x = np.random.randint(0, n)
			if x not in g_medoid_vector:
				g_medoid_vector.append(x)

		g_medoid_matrix[i] = g_medoid_vector

	return np.array(g_medoid_matrix).T

	# g_medoid_vector = []
	# while len(g_medoid_vector) < K*p:
	# 	x = np.random.randint(1,(n+1))
	# 	if x not in g_medoid_vector:
	# 		g_medoid_vector.append(x)

	# g_medoid_matrix = []
	# for i in range(p):
	# 	g_medoid_matrix.append(g_medoid_vector[(i*K):((i+1)*K)])

	# return np.array(g_medoid_matrix).T


def init_weights(K, p):
	return np.ones((K, p))

if __name__=="__main__":
	np.random.seed(42)
	print(init_gmedoid(2, 3, 9))
	print(init_weights(2, 3))
