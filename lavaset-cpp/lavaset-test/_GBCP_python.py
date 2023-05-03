# distutils: language = c++
from cmath import sqrt
import numpy as np
import sys
from sklearn import datasets 
from math import fabs
import pandas as pd
from sympy import difference_delta
import time
import sys

def quicksort(data, labels, left, right):
    pivot, fl = 0, 1
    tsd, pivot_indx = 0, 0
    ll, rr = left, right
    if left < right:
        pivot = data[left]
        while fl == 1:
            while np.all(data[ll] < pivot):
                ll += 1
            while np.all(data[rr] > pivot):
                rr -= 1
            if ll < rr:
                tsd = data[ll]
                data[ll] = data[rr]
                data[rr] = tsd

                tsd = labels[ll]
                labels[ll] = labels[rr]
                labels[rr] = tsd

                rr -= 1
            else:
                pivot_indx = rr
                fl = 0

        quicksort(data, labels, left, pivot_indx)
        quicksort(data, labels, pivot_indx+1, right)
        return data



def GBCP(M,  N,  Labels,  Data,  minleaf,  num_labels,  bcvar,  bcval):
    
    # cdef double bh, ch, gr, gl,
    # cdef int i, j, cl, nl, mj
    # cdef ndarray[double, ndim=2] sorted_data
    # cdef list diff_labels_l, diff_labels_r, diff_labels, sorted_labels

    sorted_data = np.zeros((M,N))

    diff_labels_l = list(range(num_labels))
    diff_labels_r = list(range(num_labels))
    diff_labels = list(range(num_labels))
    sorted_labels = list(range(M))

    # sorted_data = np.zeros(M)
    # bh, ch, gr, gl = 0, 0, 0, 0
    # diff_labels_l = np.zeros(num_labels, dtype=int)
    # diff_labels_r = np.zeros(num_labels, dtype=int)
    # diff_labels = np.zeros(num_labels, dtype=int)
    # sorted_labels = np.zeros(M, dtype=int)
    
    for nl in range(num_labels):
        diff_labels[nl]=0
    
    for j in range(M):
        cl = Labels[j]
        diff_labels[cl-1]+=1
    bh=0
    for nl in range(num_labels):
        bh+=diff_labels[nl]*diff_labels[nl]
    bh = 1 - (bh/(M*M))
    for i in range(N):
        for nl in range(num_labels):
            diff_labels_l[nl] = 0

            diff_labels_r[nl] = diff_labels[nl]
        
        for j in range(M):
            sorted_data[j] =  Data[j,i]#Data[i*M+j]
            sorted_labels[j] = Labels[j]
        sorted_data = quicksort(sorted_data, sorted_labels, 0, M-1)
        for mj in range(minleaf-1):
            cl=sorted_labels[mj]
            diff_labels_l[cl-1]-=1
            diff_labels_r[cl-1]+=1
            j -= 1
        
        for j in range(minleaf-1, M-minleaf):
            cl=sorted_labels[j]
            diff_labels_l[cl-1]-=1
            diff_labels_r[cl-1]+=1
            gr = 0
            gl = 0


            for nl in range(num_labels):
                gl+=diff_labels_l[nl]*diff_labels_l[nl]
                gr+=diff_labels_r[nl]*diff_labels_r[nl]
                gp = gl + gr
            gl = 1 - gl/((j+1)*(j+1))
            gr = 1 - gr/((M-j-1)*(M-j-1))
            # gp = 1 - gp/(M**2)
            ch = ((j+1)*gl/M) + ((M-j-1)*gr/M)
            if ch<bh:
                split_array = sorted_data[j+1]-sorted_data[j]
                if np.all((np.abs(split_array))>1e-15):
                # if np.abs(split_array).any()>1e-15:
                    bh=ch
                    # gini_gain = gp - bh
                    bcvar[0] = int(i) # check if it should be i ot i+1
                    bcval[0] = np.unique(0.5*(sorted_data[j+1]+sorted_data[j]))
    del diff_labels_l
    del diff_labels_r
    del diff_labels
    del sorted_labels
    del sorted_data
    return bcvar, bcval
    # free(diff_labels_l)
    # free(diff_labels_r)
    # free(diff_labels)
    # free(sorted_labels)
    # free(sorted_data)

# iris = datasets.load_iris()
# X = iris.data[47:53, 1:]  
# y = iris.target[47:53]

