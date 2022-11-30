# distutils: language=c++
import numpy as np
cimport numpy as np
cimport cython
from numpy cimport ndarray, float64_t, int_t
from libcpp cimport bool
from libc.stdlib cimport qsort
from libc.math cimport ceil, sqrt
from cython.parallel import prange, parallel


cdef ndarray[float64_t, ndim=2] get_bootstrap(ndarray[float64_t, ndim=2] data):
    return data[np.random.choice(data.shape[0], 100, replace=True), :]

cpdef bool is_pure(list Y):
    return np.unique(Y).size == 1

cpdef add_endnode(dict node, bool left, bool right):
    cdef int most_common_class
    if left:
        most_common_class = np.bincount(np.array(node['y_left']).astype('int')).argmax()
        node['left_node'] = {'end_node': True, 'y_hat': most_common_class}
        del node['y_left']
    if right:
        most_common_class = np.bincount(np.array(node['y_right']).astype('int')).argmax()
        node['right_node'] = {'end_node': True, 'y_hat': most_common_class}
        del node['y_right']

# void qsort(void *base, size_t nitems, size_t size, int (*compar)(const void *, const void*))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
#cpdef double gini_index(ndarray[float64_t, ndim=1] y_left, ndarray[float64_t, ndim=1] y_right):
cpdef double gini_index(list y_left, list y_right):
    cdef:
        double purity = 0.0
        double class_ratio
        ndarray[float64_t, ndim=2] split
        ndarray[int_t, ndim=1] class_counts, unique_classes
        int total_classes, bi
        list i 
    
    parent_node_y = y_left+y_right
    
    proportion_left = len(y_left) / len(parent_node_y)
    proportion_right = len(y_right) / len(parent_node_y)
    p_parent = (np.bincount(np.array(parent_node_y, dtype=np.int64)))/len(parent_node_y)

    p_left = (np.bincount(np.array(y_left, dtype=np.int64)))/len(y_left)
    p_right = (np.bincount(np.array(y_right, dtype=np.int64)))/len(y_right)
    gini_l = 1-np.sum(p_left**2)
    gini_r = 1-np.sum(p_right**2)
    gini_p = 1-np.sum(p_parent**2)
    
    gini_gain = gini_p - (proportion_left*gini_l + proportion_right*gini_r)
    gini = proportion_left*gini_l + proportion_right*gini_r
    
    
    return gini


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef dict get_best_split(ndarray[float64_t, ndim=2] data, ndarray[int_t, ndim=1] features, dict node_neighbors, 
                         dict loadings, dict mean, dict variance):
    cdef:
        # ndarray[double, ndim=1] y_left, y_right
        # ndarray[double, ndim=1] samples_left, samples_right
        list  df_left, df_right, b_left, b_right
        double split_point, b_split_point, gini
        double b_gini = 999.0
        int b_predictor, f_idx, predictor, row_i
        # int num_random_predictors = <int>ceil(sqrt(data.shape[1]-1)) # size of random_predictors array
        # ndarray[int_t, ndim=1] random_predictors = np.random.choice(data.shape[1]-1, num_random_predictors, replace=False)
        dict gini_dict
        ndarray[float64_t, ndim=1] loading_weight
        double node_gini
        list best_neighbors
    #print('data', data, data.shape[0])
    #print('unique', np.unique(data), np.unique(data).shape[0])

    for i_idx, f_idx in enumerate(features):
        #print(i, f_idx)
        # Split data 1. For every predictor value => For every row, to get best left / right split
        # print(split_point)
        unique = np.unique(data[:, i_idx])
        for split_point in unique:
 
            # y_left = np.empty(data.shape[0])
#             y_right = np.empty(data.shape[0])    
#             samples_left = np.empty(data.shape[0])
#             samples_right = np.empty(data.shape[0]) 
#             li = 0;
#             ri = 0;
#             for j, row in enumerate(data):
#                 if(row[i] <= split_point):
#                     y_left[li] = row[len(row)-1]
#                     samples_left[li] = int(j)
#                     li=li+1
#                 else:
#                     y_right[ri] = row[len(row)-1]
#                     samples_right[ri] = int(j)
#                     ri=ri+1
                

#             y_left = y_left[:li]
#             y_right = y_right[:ri]
#             samples_left = samples_left[:li]
#             samples_right = samples_right[:ri]
                                    
            df_left = [[int(j), row[len(row)-1]] for j, row in enumerate(data) if row[i_idx] <= split_point]
            df_right = [[int(j), row[len(row)-1]] for j, row in enumerate(data) if row[i_idx] > split_point]

            samples_left = [int(i[0]) for i in df_left]
            samples_right = [int(i[0]) for i in df_right]
        
            y_left = [i[1] for i in df_left]
            y_right = [i[1] for i in df_right]
            
            gini = gini_index(y_left, y_right)
            if gini < b_gini:
                best_neighbors = node_neighbors[f_idx]
                loading_weight = abs(loadings[f_idx])/sum(abs(loadings[f_idx]))
                gini_latent=gini*(loadings[f_idx])
                gini_dict = dict(zip(best_neighbors, gini_latent))
                b_split_point, b_predictor = split_point, f_idx
                # b_left, b_right = df_left.copy(), df_right.copy()
                #b_gini = loading_weight[0]*gini
                b_gini = gini
                best_split_dict = {
                                'feature_index': i_idx,
                                'split_point': b_split_point, 
                                'loadings': loadings[f_idx][0],
                                'mean': mean[f_idx][0], 
                                'variance': variance[f_idx][0], # taking only first value (value of the selected feature / not neighbors)
                                # 'left': b_left, 
                                # 'right': b_right, 
                                'y_left': y_left,
                                'y_right': y_right,
                                'samples_left': samples_left,  
                                'samples_right':samples_right,
                                'predictor': b_predictor, 
                                'gini': b_gini,#gain
                                'gini_latent': gini_dict
                            }
        # best_split_dict = {}

    return best_split_dict

