import numpy as np

def init_gmedoid(k_cluster, n_views, n_exemp):
	g_medoid_vector = [[]]*k_cluster
	for i in range(k_cluster):
		arr = np.array(range(n_views))
		np.random.shuffle(arr)
		g_medoid_vector[i] = arr

	return np.array(g_medoid_vector)

	# g_medoid_vector = []
	# while len(g_medoid_vector) < k_cluster*n_views:
	# 	x = np.random.randint(1,(n_exemp+1))
	# 	if x not in g_medoid_vector:
	# 		g_medoid_vector.append(x)

	# g_medoid_matrix = []
	# for i in range(n_views):
	# 	g_medoid_matrix.append(g_medoid_vector[(i*k_cluster):((i+1)*k_cluster)])

	# return np.array(g_medoid_matrix).T


def init_weights(k_cluster, n_views):
	return np.ones((k_cluster, n_views))

if __name__=="__main__":
	np.random.seed(42)
	print(init_gmedoid(2, 3, 9))
	print(init_weights(2, 3))
