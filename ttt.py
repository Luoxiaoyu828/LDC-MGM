import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels as kk


"""
row = np.array([0,1,2,0])
col = np.array([0,1,1,0])
data = np.array([1,2,4,8])
csr_matrix((data, (row, col)), shape=(3,3)).toarray()
"""
a = np.random.random([100,6])
b = kk(a,  metric='linear')