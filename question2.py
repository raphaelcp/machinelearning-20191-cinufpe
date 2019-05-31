import numpy as np
from sklearn.naive_bayes import GaussianNB
from utils.normalized_dissimilarity import load_normalized
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def main():
	L = 3 #numero de views
	K = 10 #numero de classes

	D_matrices = load_normalized(
		list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')
		# list_files=('mfeat-fac-test', 'mfeat-fou-test', 'mfeat-kar-test')
	)
	# print(D_matrices[0])

	#true_labels = np.recfromtxt('crisp_vector')
	true_labels = np.recfromtxt('data/preprocessed/true_labels.txt')

	# print(true_labels)

	prob_priori = np.zeros(K)
	for i in range(0,K):
		prob_priori[i] = np.count_nonzero(true_labels==i) / true_labels.shape[0]

	print(prob_priori)


	#GRID SEARCH CROSS VALIDATION - ENCONTRAR MELHOR NUMERO DE VIZINHOS
	kb_1 = KNeighborsClassifier()
	kb_2 = KNeighborsClassifier()
	kb_3 = KNeighborsClassifier()

	k_range = list(range(1,31))
	param_grid = dict(n_neighbors=k_range)

	grid_1 = GridSearchCV(kb_1, param_grid, cv=10, scoring='accuracy', n_jobs=4)
	grid_2 = GridSearchCV(kb_2, param_grid, cv=10, scoring='accuracy', n_jobs=4)
	grid_3 = GridSearchCV(kb_3, param_grid, cv=10, scoring='accuracy', n_jobs=4)

	grid_1.fit(D_matrices[0], true_labels)
	grid_2.fit(D_matrices[1], true_labels)
	grid_3.fit(D_matrices[2], true_labels)


	print(grid_1.best_params_['n_neighbors'])
	print(grid_2.best_params_['n_neighbors'])
	print(grid_3.best_params_['n_neighbors'])



	rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=30)

	for train_index, test_index in rskf.split(D_matrices[0], true_labels):

		# for view in range(D_matrices.shape[0]):
		matrix_train_1, matrix_test_1 = D_matrices[0][train_index], D_matrices[0][test_index]
		matrix_train_2, matrix_test_2 = D_matrices[1][train_index], D_matrices[1][test_index]
		matrix_train_3, matrix_test_3 = D_matrices[2][train_index], D_matrices[2][test_index]
		true_labels_train, true_labels_test = true_labels[train_index], true_labels[test_index]


		# print('view: ', view+1)
		# matrix = D_matrices[view]
		# print(rskf.get_n_splits(true_labels))

		gb_scores = []


			# print('TRAIN: ', len(train_index))
			# print('TEST: ', len(test_index))
		#Classificador Bayesiano Gaussiano
		gb_1 = GaussianNB()
		gb_2 = GaussianNB()
		gb_3 = GaussianNB()

		gb_1.fit(matrix_train_1, true_labels_train)
		gb_2.fit(matrix_train_2, true_labels_train)
		gb_3.fit(matrix_train_3, true_labels_train)

		pred_1 = gb_1.predict_proba(matrix_test_1)
		pred_2 = gb_2.predict_proba(matrix_test_2)
		pred_3 = gb_3.predict_proba(matrix_test_3)

		print(pred_2.shape)
		gb_ensemble = np.argmax(((1-L)*(prob_priori) + pred_1 + pred_2 + pred_3), axis=1)
		# print(((1-L)*(prob_priori) + pred_1 + pred_2 + pred_3))

		score = np.equal(gb_ensemble, true_labels_test).sum() / true_labels_test.shape[0]
		print("%.5f"%score, end=' - ');

		kb_1 = KNeighborsClassifier(n_neighbors=grid_1.best_params_['n_neighbors'])
		kb_2 = KNeighborsClassifier(n_neighbors=grid_2.best_params_['n_neighbors'])
		kb_3 = KNeighborsClassifier(n_neighbors=grid_3.best_params_['n_neighbors'])

		kb_1.fit(matrix_train_1, true_labels_train)
		kb_2.fit(matrix_train_2, true_labels_train)
		kb_3.fit(matrix_train_3, true_labels_train)

		pred_1 = kb_1.predict_proba(matrix_test_1)
		pred_2 = kb_2.predict_proba(matrix_test_2)
		pred_3 = kb_3.predict_proba(matrix_test_3)

		kb_ensemble = np.argmax(((1-L)*(prob_priori) + pred_1 + pred_2 + pred_3), axis=1)
		print(((1-L)*(prob_priori) + pred_1 + pred_2 + pred_3))

		#score dos ensembles
		score = np.equal(kb_ensemble, true_labels_test).sum() / true_labels_test.shape[0]
		print("%.5f"%score);


		# print(gb_ensemble); exit()

			# gb_score = gb.score(matrix_test, true_labels_test)
			# gb_scores.append(gb_score)

		# print(len(gb_scores), np.mean(gb_scores), np.std(gb_scores))

if __name__ == '__main__':
	main()

