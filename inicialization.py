import numpy as np 

def init_gmedoid(k_cluster, n_views, n_exemp):
	g_medoid = []
	while len(g_medoid) < k_cluster*n_views:
		x = np.random.randint(1,(n_exemp+1))
		if x not in g_medoid:
			g_medoid.append(x)

	matriz = []
	for i in range(n_views):
		matriz.append(g_medoid[(i*k_cluster):((i+1)*k_cluster)])	

	# g_medoid_1 = g_medoid[:10]
	# g_medoid_2 = g_medoid[10:20]
	# g_medoid_3 = g_medoid[20:30]
	return matriz


def init_weights(k_cluster, n_views):
	
	return [[1]*k_cluster]*n_views





if __name__=="__main__":
	print(init_gmedoid(10, 3, 2000))
	print(init_weights(10,3))