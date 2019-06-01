import numpy as np
import time
from mvfcmddv import MVFCMddV
from utils.convert_crisp import convert_crisp
from utils.normalized_dissimilarity import normalized_dissimilarity
from utils.convert_crisp import convert_crisp
from utils.rand_index import *
from utils.crisp_obj import *

def min_repeat(arr, K=10):
	counter = np.zeros(K, dtype=int)
	for item in arr:
		counter[item] += 1
	return counter.min()

def save_result(obj, it):
	np.savetxt('output/%s-G'%(it), obj['G'], fmt='%i')
	np.savetxt('output/%s-W'%(it), obj['W'], fmt='%.7f')
	np.savetxt('output/%s-U'%(it), obj['U'], fmt='%.7f')
	np.savetxt('output/%s-performance'%(it), [obj['performance']], fmt='%.7f')

def main():

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
		save_result(out, '%02d'%(i))

		crisp = convert_crisp(out['U'])
		crisp_vector = new_crisp(crisp)
		if min_repeat(crisp_vector) < 10:
			print('nao rolou')
			continue
		else:
			print('rolouu')

		if len(best_out) == 0 or out['performance'] < best_out['performance']:
			best_out = out.copy()
			save_result(best_out, 'best')

	crisp = convert_crisp(best_out['U'])
	crisp_vector = new_crisp(crisp)
	print(crisp_obj(crisp_vector, K))
	crisp_true = true_labels(n, K)
	# np.savetxt('crisp_true',crisp_true, fmt='%i')
	np.savetxt('crisp_vector',crisp_vector, fmt='%i')
	print(crisp_vector)
	print('Indice de Rand Corrigido', adjusted_rand_score(crisp_vector, crisp_true))

if __name__ == '__main__':
	np.random.seed(42)
	main()
