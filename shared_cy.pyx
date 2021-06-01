import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt, log

##### Cython functions for EM algorithm in HaploNet #####
@boundscheck(False)
@wraparound(False)
cpdef emLoop(float[:,:,::1] L, float[:,:,::1] F, float[:,::1] Q, \
				float[:,:,::1] Fnew, float[:,:,::1] Qnew, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int K = Q.shape[1]
	cdef int w, d, i, a, c, k
	cdef float pSum, postKY, sumY
	with nogil:
		for w in prange(W, num_threads=t):
			# Reset Fnew
			for k in range(K):
				for c in range(C):
					Fnew[w, k, c] = 0.0

			# Reset Qnew
			for d in range(N//2):
				for k in range(K):
					Qnew[w, d, k] = 0.0

			# Run EM
			for i in range(0, N, 2):
				for a in range(2):
					pSum = 0.0
					for k in range(K):
						for c in range(C):
							pSum = pSum + L[w, i+a, c]*F[w, k, c]*Q[i//2, k]
					for k in range(K):
						for c in range(C):
							if pSum > 1e-7:
								postKY = L[w, i+a, c]*F[w, k, c]*Q[i//2, k]/pSum
								Fnew[w, k, c] += postKY
								Qnew[w, i//2, k] += postKY

			# Update F
			for k in range(K):
				sumY = 0.0
				for c in range(C):
					sumY = sumY + Fnew[w, k, c]
				for c in range(C):
					F[w, k, c] = Fnew[w, k, c]/sumY


# Q update
@boundscheck(False)
@wraparound(False)
cpdef updateQ(float[:,:,::1] Qnew, float[:,::1] Q, int t):
	cdef int W = Qnew.shape[0]
	cdef int N = Qnew.shape[1]
	cdef int K = Qnew.shape[2]
	cdef int w, i, k
	with nogil:
		for i in prange(N, num_threads=t):
			for k in range(K):
				Q[i, k] = 0.0
				for w in range(W):
					Q[i, k] += Qnew[w, i, k]
			for k in range(K):
				Q[i, k] /= 2.0*float(W)


# Log-likelihood
@boundscheck(False)
@wraparound(False)
cpdef logLike(float[:,:,::1] L, float[:,:,::1] F, float[:,::1] Q, \
				float[::1] logVec, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int K = Q.shape[1]
	cdef int w, i, a, c, k
	cdef float pLike
	with nogil:
		for w in prange(W, num_threads=t):
			logVec[w] = 0.0
			for i in range(0, N, 2):
				for a in range(2):
					pLike = 0.0
					for k in range(K):
						for c in range(C):
							pLike = pLike + L[w, i+a, c]*F[w, k, c]*Q[i//2, k]
					logVec[w] += log(pLike)


### Acceleration functions
# Minus Q
@boundscheck(False)
@wraparound(False)
cpdef matMinusQ(float[:,::1] Q1, float[:,::1] Q2, float[:,::1] diffQ):
	cdef int N = Q1.shape[0]
	cdef int K = Q1.shape[1]
	cdef int i, k
	for i in range(N):
		for k in range(K):
			diffQ[i, k] = Q1[i, k] - Q2[i, k]

# SumSquare Q
@boundscheck(False)
@wraparound(False)
cpdef matSumSquareQ(float[:,::1] diffQ):
	cdef int N = diffQ.shape[0]
	cdef int K = diffQ.shape[1]
	cdef int i, k
	cdef float sumQ = 0.0
	for i in range(N):
		for k in range(K):
			sumQ += diffQ[i, k]*diffQ[i, k]
	return sumQ

# Update F alpha
@boundscheck(False)
@wraparound(False)
cpdef accelUpdateQ(float[:,::1] Q, float[:,::1] Q0, float[:,::1] diff1, \
					float[:,::1] diff3, float alpha, int t):
	cdef int N = Q.shape[0]
	cdef int K = Q.shape[1]
	cdef int i, k
	cdef float sumK
	with nogil:
		for i in prange(N, num_threads=t):
			sumK = 0.0
			for k in range(K):
				Q[i, k] = Q0[i, k] + 2*alpha*diff1[i, k] + \
							alpha*alpha*diff3[i, k]
				Q[i, k] = min(max(Q[i, k], 1e-7), 1-(1e-7))
				sumK = sumK + Q[i, k]
			for k in range(K):
				Q[i, k] /= sumK

# Minus F
@boundscheck(False)
@wraparound(False)
cpdef matMinusF(float[:,:,::1] F1, float[:,:,::1] F2, float[:,:,::1] diffF):
	cdef int W = F1.shape[0]
	cdef int K = F1.shape[1]
	cdef int C = F1.shape[2]
	cdef int w, k, c
	for w in range(W):
		for k in range(K):
			for c in range(C):
				diffF[w, k, c] = F1[w, k, c] - F2[w, k, c]

# SumSquare F
@boundscheck(False)
@wraparound(False)
cpdef matSumSquareF(float[:,:,::1] diffF):
	cdef int W = diffF.shape[0]
	cdef int K = diffF.shape[1]
	cdef int C = diffF.shape[2]
	cdef int w, k, c
	cdef float sumF = 0.0
	for w in range(W):
		for k in range(K):
			for c in range(C):
				sumF += diffF[w, k, c]*diffF[w, k, c]
	return sumF

# Update F alpha
@boundscheck(False)
@wraparound(False)
cpdef accelUpdateF(float[:,:,::1] F, float[:,:,::1] F0, float[:,:,::1] diff1, \
					float[:,:,::1] diff3, float alpha, int t):
	cdef int W = F.shape[0]
	cdef int K = F.shape[1]
	cdef int C = F.shape[2]
	cdef int w, k, c
	cdef float sumY
	with nogil:
		for w in prange(W, num_threads=t):
			for k in range(K):
				sumY = 0.0
				for c in range(C):
					F[w, k, c] = F0[w, k, c] + 2*alpha*diff1[w, k, c] + \
									alpha*alpha*diff3[w, k, c]
					F[w, k, c] = min(max(F[w, k, c], 1e-7), 1-(1e-7))
					sumY = sumY + F[w, k, c]
				for c in range(C):
					F[w, k, c] /= sumY


##### Cython functions for PCA in HaploNet #####
# Standardize cluster matrix
@boundscheck(False)
@wraparound(False)
cpdef standardizeY(signed char[:,::1] L, float[::1] F, int t):
	cdef int N = L.shape[0]
	cdef int S = L.shape[1]
	cdef int i, s
	with nogil:
		for i in prange(N, num_threads=t):
			for s in range(S):
				L[i, s] = (L[i, s] - 2*F[s])/sqrt(2*F[s]*(1 - F[s]))

# Standardize unphased cluster matrix
@boundscheck(False)
@wraparound(False)
cpdef standardizeY_unphased(signed char[:,::1] L, float[::1] F, int t):
	cdef int N = L.shape[0]
	cdef int S = L.shape[1]
	cdef int i, s
	with nogil:
		for i in prange(N, num_threads=t):
			for s in range(S):
				L[i, s] = (L[i, s] - F[s])/sqrt(F[s]*(1 - F[s]))

# Covariance estimation
@boundscheck(False)
@wraparound(False)
cpdef covarianceY(signed char[:,::1] L, float[::1] F, float[:,::1] C, int t):
	cdef int N = L.shape[0]
	cdef int S = L.shape[1]
	cdef int i, j, s
	with nogil:
		for i in prange(N, num_threads=t):
			for j in range(i, N):
				for s in range(S):
					C[i, j] += (L[i, s] - 2*F[s])*(L[j, s] - 2*F[s])/ \
								(2*F[s]*(1 - F[s]))
				C[i, j] /= float(S)
				C[j, i] = C[i, j]

# Covariance estimation - unphased
@boundscheck(False)
@wraparound(False)
cpdef covarianceY_unphased(signed char[:,::1] L, float[::1] F, float[:,::1] C, int t):
	cdef int N = L.shape[0]
	cdef int S = L.shape[1]
	cdef int i, j, s
	with nogil:
		for i in prange(N, num_threads=t):
			for j in range(i, N):
				for s in range(S):
					C[i, j] += (L[i, s] - F[s])*(L[j, s] - F[s])/ \
								(F[s]*(1 - F[s]))
				C[i, j] /= float(S)
				C[j, i] = C[i, j]
