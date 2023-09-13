# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, log

# Estimate mean within population distance
cpdef float withinDist(float[:,::1] Q, int[::1] L, int pop):
    cdef int N = Q.shape[0]
    cdef int K = Q.shape[1]
    cdef int i, j, k
    cdef int c = 0
    cdef float tmp
    cdef float sum = 0.0
    for i in range(N):
        if L[i] == pop:
            for j in range(i+1, N):
                if L[j] == pop:
                    c = c + 1
                    tmp = 0.0
                    for k in range(K):
                        tmp = tmp + (Q[i, k] - Q[j, k])*(Q[i, k] - Q[j, k])
                    sum = sum + sqrt(max(tmp, 1e-6))
    return sum/float(c)

# Estimate mean between population distance
cpdef float betweenDist(float[:,::1] Q, int[::1] L, int pop):
    cdef int N = Q.shape[0]
    cdef int K = Q.shape[1]
    cdef int i, j, k
    cdef int c = 0
    cdef float tmp
    cdef float sum = 0.0
    for i in range(N):
        if L[i] == pop:
            for j in range(N):
                if L[j] != pop:
                    c = c + 1
                    tmp = 0.0
                    for k in range(K):
                        tmp = tmp + (Q[i, k] - Q[j, k])*(Q[i, k] - Q[j, k])
                    sum = sum + sqrt(max(tmp, 1e-6))
    return sum/float(c)
