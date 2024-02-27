
from libcpp.vector cimport vector

cdef extern from "quickie.h":
    cdef cppclass TLabel:
        pass

    void quicksort(double *data, TLabel *labels, int left, int right)

cdef extern from "GBCP.h":
    void GBCP(int M, int N, double* Labels, double* Data, int minleaf, int num_labels, double* bcvar, double* bcval)


def gbc_p(int M, int N, double[:] Labels, double[:,:] Data, int minleaf, int num_labels):
    cdef double bcvar[1]
    cdef double bcval[1]

    GBCP(M, N, &Labels[0], &Data[0,0], minleaf, num_labels, bcvar, bcval)

    return bcvar[0], bcval[0]

