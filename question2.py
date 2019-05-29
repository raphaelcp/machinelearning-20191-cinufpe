import numpy as np
from sklearn.naive_bayes import GaussianNB
from utils.normalized_dissimilarity import normalized_dissimilarity
from sklearn.model_selection import RepeatedStratifiedKFold

def main():
	D_matrices = normalized_dissimilarity(
		list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')
		# list_files=('mfeat-fac-test', 'mfeat-fou-test', 'mfeat-kar-test')
	)

	true_labels = np.recfromtxt('data/preprocessed/true_labels.txt')

	for view in range(D_matrices.shape[0]):
		print('view: ', view+1)
		matrix = D_matrices[view]
		rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30)
		# print(rskf.get_n_splits(true_labels))

		gb_scores = []
		for train_index, test_index in rskf.split(matrix, true_labels):

			matrix_train, matrix_test = matrix[train_index], matrix[test_index]
			true_labels_train, true_labels_test = true_labels[train_index], true_labels[test_index]

			# print('TRAIN: ', len(train_index))
			# print('TEST: ', len(test_index))
		#Classificador Bayesiano Gaussiano
			gb = GaussianNB()
			gb.fit(matrix_train, true_labels_train)
			gb_score = gb.score(matrix_test, true_labels_test)
			gb_scores.append(gb_score)
			print(gb.get_params)

		print(len(gb_scores), np.mean(gb_scores), np.std(gb_scores))

if __name__ == '__main__':
	main()
