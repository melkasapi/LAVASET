
import numpy as np
import lavaset.cython_wrapper as cython_wrapper
# from lavaset.cython_wrapper import gbc_p
import pandas as pd
# from _GBCP_python import GBCP

def best_cut_node(method, Data, Labels, minleaf, max_label):
    
    M, N = Data.shape # following the cpp code nomenclature
    
    # N = int(max(1, int(np.sqrt(N))))

    bcvar = np.zeros(1, dtype=np.double)
    bcvar[0]=-1

    bcval = np.zeros(1)
    
    if ((method[0] == 'c')  or (method[0] == 'g')): 
        num_labels = max_label
    if (method[0] == 'g'):
        bcvar, bcval = cython_wrapper.gbc_p(M, N, Labels, Data, minleaf, num_labels)#, bcvar, bcval)
    
    del method
    return int(bcvar), bcval#int(bcvar[0]), bcval[0]

# iris = datasets.load_iris()
# X = iris.data[45:55]  
# y = np.array(iris.target[45:55],dtype=np.double)#.tolist()


# nmr_peaks = pd.read_csv('~/Documents/IBS/NMR_data/IBS_HNMR_data_n267.csv')

# # y = np.array(nmr_peaks.iloc[:, 1])


# X = np.array(nmr_peaks.iloc[:, 1500:1750], dtype=np.double)
# y = np.array(nmr_peaks.iloc[:, 1], dtype=np.int)

# start = time.time()
# for i in range(20):
    
#     X = np.array(nmr_peaks.iloc[:, i+100:i+450], dtype=np.double)
#     y = np.array(nmr_peaks.iloc[:, 1], dtype=np.int)     
#     print(best_cut_node('g', X, y, minleaf=1, max_label=2))
#     print(i)
# end = time.time()
# print(end-start)

