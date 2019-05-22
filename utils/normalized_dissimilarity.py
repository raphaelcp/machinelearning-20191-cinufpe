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

		# aplica a normalização
		data = data.astype(float)
		data_scaled = minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)

		# calcula a matriz de dissimilaridade
		df_scale_dist = pdist(data_scaled, metric='euclidean')
		df_scale_dist = squareform(df_scale_dist) # coloca em formato de matriz n*n
		D_matrices.append(df_scale_dist)

	return np.array(D_matrices)

if __name__ == "__main__":
	print(normalized_dissimilarity(list_files=('mfeat-fac-test', 'mfeat-fou-test', 'mfeat-kar-test')))
# # calcula a matriz de dissimilaridade
# df_scale_dist = pdist(df_scale.values, metric='euclidean')
# df_scale_dist = squareform(df_scale_dist) # coloca em formato de matriz n*n


# #Carregamento de Arquivos
# data_fac = np.recfromtxt('mfeat-fac')
# data_fou = np.recfromtxt('mfeat-fou')
# data_kar = np.recfromtxt('mfeat-kar')


# #Normalização para intervalo de 0<x<1
# scaler = MinMaxScaler()

# data_fac = scaler.fit(data_fac)
# data_fou = scaler.fit(data_fou)
# data_kar = scaler.fit(data_kar)

# data_fac = scaler.transform(data_fac)
# data_fou = scaler.transform(data_fou)
# data_kar = scaler.transform(data_kar)


