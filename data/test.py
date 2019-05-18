import numpy as np

for file in ('mfeat-fac', 'mfeat-fou', 'mfeat-kar'):
	data = np.recfromtxt(file)
	data_test = []

	n = len(data)

	for i in range(0, n, 200):
		for j in range(10):
			data_test.append(data[i+j])

	print(np.array(data_test).shape)

	# np.array(data_test)
	np.savetxt(file + '-test', data_test, delimiter=",")
