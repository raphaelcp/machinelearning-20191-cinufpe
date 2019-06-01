import numpy as np
from sklearn.preprocessing import scale, minmax_scale
from scipy.spatial.distance import pdist, squareform

# Normailzação dos dados e matriz de dissimilaridade
def normalized_dissimilarity(list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')):
	D_matrices = []
	for file in list_files:
		print(file)

		# carrega os dados
		data = np.recfromtxt('data/' + file)
		if len(data.shape) == 1:
			data = data.reshape(data.shape[0], 1)

		# aplica a normalização
		data = data.astype(float)
		data_scaled = minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)

		# calcula a matriz de dissimilaridade
		df_scale_dist = pdist(data_scaled, metric='euclidean')
		df_scale_dist = squareform(df_scale_dist) # coloca em formato de matriz n*n
		D_matrices.append(df_scale_dist)

	return np.array(D_matrices)

def load_normalized(list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')):
	D_matrices = []
	for file in list_files:
		print(file)

		# carrega os dados
		data = np.recfromtxt('data/' + file)
		if len(data.shape) == 1:
			data = data.reshape(data.shape[0], 1)

		# aplica a normalização
		data = data.astype(float)
		data_= minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)
		D_matrices.append(data)
	return D_matrices

if __name__ == "__main__":
	#print(normalized_dissimilarity(list_files=('mfeat-fac-test', 'mfeat-fou-test', 'mfeat-kar-test')))
	print(load_normalized(list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')))