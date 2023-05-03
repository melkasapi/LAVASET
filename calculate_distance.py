import numpy as np
from sklearn.neighbors import KDTree

def knn_calculation(data, neigh_number):
    if data.ndim == 1:
        X = data.to_numpy(dtype=float)
        X = np.append([X], [data.to_numpy(dtype=float)], axis=0)
        kdtree = KDTree(X.T)
        points = kdtree.query(X.T,neigh_number+1)[1]
        return points 
    else:
        X = data.to_numpy(dtype=float)
        kdtree = KDTree(X.T)
        points = kdtree.query(X.T,neigh_number+1)[1]
        return points 