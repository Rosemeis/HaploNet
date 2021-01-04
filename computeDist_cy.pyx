import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt, log, fabs

##### Cython functions for covariance/distance estimation in HaploNet #####
# Euclidean distance
@boundscheck(False)
@wraparound(False)
cpdef EuclideanMatrix(float[:,:,::1] Z, float[:,::1] D, int t):
	cdef int N = Z.shape[0]
	cdef int M = Z.shape[1]
	cdef int K = Z.shape[2]
	cdef int i, j, m, k, a, b
	cdef float tmp1, tmp2
	with nogil:
		for i in prange(0, N, 2, num_threads=t):
			for j in range(i, N, 2):
				for m in range(M):
					tmp1 = 0.0
					for a in range(2):
						for b in range(2):
							tmp2 = 0.0
							for k in range(K):
								tmp2 += (Z[i+a,m,k] - Z[j+b,m,k])**2
							tmp1 += sqrt(tmp2)
					D[i//2,j//2] += tmp1
				D[i//2,j//2] /= float(4*M)
				D[j//2,i//2] = D[i//2,j//2]

# Mahalanobis distance
@boundscheck(False)
@wraparound(False)
cpdef MahalanobisMatrix(float[:,:,::1] Z, float[:,:,::1] eV, float[:,::1] D, int t):
	cdef int N = Z.shape[0]
	cdef int M = Z.shape[1]
	cdef int K = Z.shape[2]
	cdef int i, j, m, k, a, b
	cdef float tmp1, tmp2, comb
	with nogil:
		for i in prange(0, N, 2, num_threads=t):
			for j in range(i, N, 2):
				for m in range(M):
					tmp1 = 0.0
					for a in range(2):
						for b in range(2):
							tmp2 = 0.0
							for k in range(K):
								comb = (eV[i+a,m,k] + eV[j+b,m,k])/2.0
								tmp2 += ((Z[i+a,m,k] - Z[j+b,m,k])**2)/comb
							tmp1 += sqrt(tmp2)
					D[i//2,j//2] += tmp1
				D[i//2,j//2] /= float(4*M)
				D[j//2,i//2] = D[i//2,j//2]

# Bhattacharyya distance
@boundscheck(False)
@wraparound(False)
cpdef BhattacharyyaMatrix(float[:,:,::1] Z, float[:,:,::1] V, float[:,:,::1] eV, float[:,::1] D, int t):
	cdef int N = Z.shape[0]
	cdef int M = Z.shape[1]
	cdef int K = Z.shape[2]
	cdef int i, j, m, k, a, b
	cdef float tmp1, tmp2, sum1, sum2, sum3, comb
	with nogil:
		for i in prange(0, N, 2, num_threads=t):
			for j in range(i, N, 2):
				for m in range(M):
					tmp1 = 0.0
					for a in range(2):
						for b in range(2):
							tmp2 = 0.0
							sum1 = 0.0
							sum2 = 0.0
							sum3 = 0.0
							for k in range(K):
								comb = (eV[i+a,m,k] + eV[j+b,m,k])/2.0
								tmp2 += ((Z[i+a,m,k] - Z[j+b,m,k])**2)/comb
								sum1 += V[i+a,m,k]
								sum2 += V[j+b,m,k]
								sum3 += log(comb)
							tmp1 += 0.125*(tmp2) + 0.5*(sum3 - 0.5*(sum1 + sum2))
					D[i//2,j//2] += tmp1
				D[i//2,j//2] /= float(4*M)
				D[j//2,i//2] = D[i//2,j//2]

# Manhattan distance
@boundscheck(False)
@wraparound(False)
cpdef ManhattanMatrix(float[:,:,::1] Y, float[:,::1] D, int t):
	cdef int N = Y.shape[0]
	cdef int M = Y.shape[1]
	cdef int K = Y.shape[2]
	cdef int i, j, m, k, a, b
	cdef float tmp
	with nogil:
		for i in prange(0, N, 2, num_threads=t):
			for j in range(i, N, 2):
				for m in range(M):
					tmp = 0.0
					for a in range(2):
						for b in range(2):
							for k in range(K):
								tmp += fabs(Y[i+a,m,k] - Y[j+b,m,k])
					D[i//2,j//2] += tmp
				D[i//2,j//2] /= float(4*M)
				D[j//2,i//2] = D[i//2,j//2]

			for m in range(M):
				tmp = 0.0
				for k in range(K):
					tmp += fabs(Y[i+a,m,k] - Y[j+b,m,k])


# Covariance matrix
@boundscheck(False)
@wraparound(False)
cpdef CovarianceCluster(float[:,:,::1] Z, float[:,:] F, float[:,::1] C, int t):
	cdef int N = Z.shape[0]
	cdef int M = Z.shape[1]
	cdef int K = Z.shape[2]
	cdef int i, j, m, k, a, b
	cdef float tmp
	with nogil:
		for i in prange(0, N, 2, num_threads=t):
			for j in range(i, N, 2):
				for m in range(M):
					tmp = 0.0
					for a in range(2):
						for b in range(2):
							for k in range(K):
								tmp += (Z[i+a,m,k] - F[m,k])*(Z[j+b,m,k] - F[m,k])
					C[i//2,j//2] += tmp
				C[i//2,j//2] /= float(K*4*M)
				C[j//2,i//2] = C[i//2,j//2]
