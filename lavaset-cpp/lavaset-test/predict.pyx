cimport numpy as np
import numpy as np

cpdef tree_predict(np.ndarray[np.double_t, ndim=2] Data, np.ndarray[np.double_t, ndim=1] cut_var, 
                np.ndarray[np.double_t, ndim=1] cut_val, np.ndarray[np.double_t, ndim=1] nodechilds, 
                np.ndarray[np.double_t, ndim=1] nodelabel):
    cdef int i, current_node, M, cvar
    cdef np.ndarray[np.double_t, ndim=1] tree_output
    
    M = Data.shape[0]
    tree_output = np.zeros(M, dtype=np.double)

    for i in range(M):
        current_node = 0
        while nodechilds[current_node] != 0:
            cvar = <int>cut_var[current_node]
            if Data[i, cvar-1] < cut_val[current_node]:
                current_node = <int>nodechilds[current_node] - 1
            else:
                current_node = <int>nodechilds[current_node]
        tree_output[i] = nodelabel[current_node]

    return tree_output
