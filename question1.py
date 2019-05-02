# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt


def calculate_dissimilarity(data):
    y = np.array([([np.linalg.norm(x1 - x2) for x2 in data]) for x1 in data], dtype=np.float32)
    return y

def calculate_weight_data(data, weight, sqrt=False):
 
    if sqrt:
        data = data ** 2

    if weight is None:
        return data
    else:
        assert weight.shape[0] == data.shape[0]
        return np.matmul(data, weight)


def unique_rows(a, return_index=False, return_inverse=False, return_counts=False):
    try:
        dummy, uniqi, inv_uniqi, counts = np.unique(a.view(a.dtype.descr * a.shape[1]), return_index=True, return_inverse=True, return_counts=True)
        out = [a[uniqi, :]]
        if return_index:
            out.append(uniqi)
        if return_inverse:
            out.append(inv_uniqi)
        if return_counts:
            out.append(counts)
    except ValueError:
        s = set()
        for i in range(a.shape[0]):
            s.add(tuple(a[i, :].tolist()))
        out = [np.array([row for row in s])]
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

def calculate_membership(data, medoids, param=2):

    r = data[:, medoids]
  
    assert param >= 1
    tmp = (1 / r) ** (1 / (param - 1))

    membership = tmp / tmp.sum(axis=1, keepdims=True)
    for medi, med in enumerate(medoids):
        membership[med, :] = 0.
        membership[med, medi] = 1.
    return membership


def fuzzy_c_medoids(data, c, weights=None, steps=1, max_iter=1000, init_idxs=None, possible_medoid_idx=None):
    
    N = data.shape[0]

    if init_idxs is None:
        init_idxs = np.arange(N)

    w_data = calculate_weight_data(data, weights)
    
    if not possible_medoid_idx is None:
        init_idxs = np.array([i for i in init_idxs if i in possible_medoid_idx], dtype=int)
    else:
        possible_medoid_idx = np.arange(N)

    if len(init_idxs) == 0:
        print('Error init init_idxs')
        return

    all_medoids = np.zeros((steps, c))
    best_rest = None
    
    for step in range(steps):
    
        current_medoids = np.random.permutation(init_idxs)[:c]
        new_medoids = np.zeros(c, dtype=int)
        for i in range(max_iter):
    
            membership = calculate_membership(data, current_medoids, param=2)
            
            accumulated_U = 0
            
            for medi, _ in enumerate(current_medoids):
                U_Mat = np.tile(membership[:, medi][:, None].T, (len(possible_medoid_idx), 1)) * w_data[possible_medoid_idx, :]
                U_Vec = U_Mat.sum(axis=1)
                min_idx = np.argmin(U_Vec)
                new_medoids[medi] = possible_medoid_idx[min_idx]
                
                accumulated_U += U_Vec[min_idx]
                
            if (new_medoids == current_medoids).all():
                all_medoids[step, :] = sorted(current_medoids)
                break
            current_medoids = new_medoids.copy()
            
        if best_rest is None or accumulated_U < best_rest:
            best_rest = accumulated_U
            best_medoids = current_medoids.copy()
            best_membership = membership.copy()
            best_N_iter = i + 1
    
    N_found = unique_rows(all_medoids).shape[0]
    return best_medoids, best_membership, best_N_iter, N_found


def test_MVFCMddV(steps=20, c=10, max_iter=1000):
    from sklearn.preprocessing import MinMaxScaler
    
    path = ''
    ar = np.loadtxt("{}{}".format(path, "mfeat-fac"))
    data = np.array(ar, dtype=np.float32)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    diss = calculate_dissimilarity(data)
    
    results = fuzzy_c_medoids(diss, c=c, max_iter=max_iter, steps=steps)
    
    return diss, results
#-------------------------------------------

data, (best_medoids, best_membership, best_N_iter, N_found) = test_MVFCMddV()
print (best_medoids, best_membership, best_N_iter, N_found)

print (best_membership.shape)
print (type(best_membership))

best_membership=np.transpose(best_membership)

print (best_membership.shape)

# Show 3-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model')
for j in range(10):
    print (data[0, best_membership.argmax(axis=0)])
    print (data[1, best_membership.argmax(axis=0)])
            
    ax2.plot(data[0, best_membership.argmax(axis=0) == j],
             data[1, best_membership.argmax(axis=0) == j], 'o',
             label='clusters ' + str(j))
ax2.legend()

plt.show()

