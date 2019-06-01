import numpy as np
import time
from mvfcmddv import MVFCMddV
from utils.convert_crisp import convert_crisp
from utils.normalized_dissimilarity import normalized_dissimilarity
from utils.convert_crisp import convert_crisp
from sklearn.metrics.cluster import adjusted_rand_score
from utils.rand_index import *
from utils.crisp_obj import *
import os

def min_repeat(arr, K=10):
	counter = np.zeros(K, dtype=int)
	for item in arr:
		counter[item] += 1
	return counter.min()

def main():
	path = 'output/question-1'
	os.path.exists(path) or os.makedirs(path)

	D_matrices = normalized_dissimilarity(
		list_files=('mfeat-fac', 'mfeat-fou', 'mfeat-kar')
		# list_files=('mfeat-fac-test-2', 'mfeat-fou-test-2', 'mfeat-kar-test-2')
		# list_files=(
		# 	'ecoli/ecoli.data-1',
		# 	'ecoli/ecoli.data-2',
		# 	'ecoli/ecoli.data-3',
		# 	'ecoli/ecoli.data-4',
		# 	'ecoli/ecoli.data-5',
		# 	'ecoli/ecoli.data-6',
		# 	'ecoli/ecoli.data-7',
		# )
	)
	n = D_matrices[0].shape[0]
	K = 10
	best_out = {}
	for i in range(100):
		print('# run %d'%(i+1))
		out = MVFCMddV(D_matrices, K)

		crisp = convert_crisp(out['U'])
		crisp_vector = new_crisp(crisp)
		if min_repeat(crisp_vector) >= 10 and (len(best_out) == 0 or out['performance'] < best_out['performance']):
			best_out = out.copy()
			np.savetxt('%s/best-G'%(path), best_out['G'], fmt='%i')
			np.savetxt('%s/best-W'%(path), best_out['W'], fmt='%.7f')
			np.savetxt('%s/best-U'%(path), best_out['U'], fmt='%.7f')
			np.savetxt('%s/best-J'%(path), best_out['J'], fmt='%.7f')
			np.savetxt('%s/best-performance'%(path), [best_out['performance']], fmt='%.7f')

	crisp = convert_crisp(best_out['U'])
	np.savetxt('%s/crisp'%(path), crisp, fmt='%i')

	crisp_vector = new_crisp(crisp)
	np.savetxt('%s/crisp-vector'%(path), crisp_vector, fmt='%i')

	crisp_arr_obj = crisp_obj(crisp_vector, K)
	f = open('%s/crisp-obj'%(path), 'w+')
	for x in crisp_arr_obj:
		f.write(str(x) + '\n')
	f.close()

	crisp_true = true_labels(n, K)
	np.savetxt('%s/crisp-true'%(path), crisp_true, fmt='%i')

	corrected_rand_index = adjusted_rand_score(crisp_vector, crisp_true)
	print('Indice de Rand Corrigido: %.7f'%(corrected_rand_index))
	np.savetxt('%s/corrected-rand-index'%(path), [corrected_rand_index], fmt='%.7f')

if __name__ == '__main__':
	np.random.seed(42)
	main()
