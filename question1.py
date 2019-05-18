import numpy as np
from utils.normalized_dissimilarity import normalized_dissimilarity


def main():
	D_matrices = normalized_dissimilarity(
		list_files=('mfeat-fac-test', 'mfeat-fou-test', 'mfeat-kar-test')
	)
	print(D_matrices)


if __name__ == '__main__':
	main()
