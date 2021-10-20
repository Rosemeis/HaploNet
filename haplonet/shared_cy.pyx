# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt, log, exp

##### Cython functions for EM algorithm in HaploNet #####
# Convert loglikes to normalized likes
cpdef createLikes(float[:,:,::1] L, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int w, i, c
	cdef float tmpMax, tmpSum
	with nogil:
		for w in prange(W, num_threads=t):
			for i in range(N):
				tmpMax = L[w, i, 0]
				tmpSum = 0.0
				for c in range(1, C):
					if L[w, i, c] > tmpMax:
						tmpMax = L[w, i, c]
				for c in range(C):
					L[w, i, c] = L[w, i, c] - tmpMax
					L[w, i, c] = exp(L[w, i, c])


# Main EM iteration
cpdef emLoop(float[:,:,::1] L, double[:,:,::1] F, double[:,::1] Q, \
				double[:,:,::1] Fnew, double[:,:,::1] Qnew, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int K = Q.shape[1]
	cdef int w, d, i, a, c, k
	cdef double pSum, postKY, sumY
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
cpdef updateQ(double[:,:,::1] Qnew, double[:,::1] Q, int t):
	cdef int W = Qnew.shape[0]
	cdef int N = Qnew.shape[1]
	cdef int K = Qnew.shape[2]
	cdef int w, i, k
	cdef double sumQ
	with nogil:
		for i in prange(N, num_threads=t):
			sumQ = 0.0
			for k in range(K):
				Q[i, k] = 0.0
				for w in range(W):
					Q[i, k] += Qnew[w, i, k]
			for k in range(K):
				Q[i, k] /= 2.0*float(W)
				Q[i, k] = min(max(Q[i, k], 1e-7), 1-(1e-7))
				sumQ = sumQ + Q[i, k]
			for k in range(K):
				Q[i, k] = Q[i, k]/sumQ


# Log-likelihood
cpdef logLike(float[:,:,::1] L, double[:,:,::1] F, double[:,::1] Q, \
				double[::1] logVec, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int K = Q.shape[1]
	cdef int w, i, a, c, k
	cdef double pLike
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
cpdef matMinusQ(double[:,::1] Q1, double[:,::1] Q2, double[:,::1] diffQ):
	cdef int N = Q1.shape[0]
	cdef int K = Q1.shape[1]
	cdef int i, k
	for i in range(N):
		for k in range(K):
			diffQ[i, k] = Q1[i, k] - Q2[i, k]

# SumSquare Q
cpdef matSumSquareQ(double[:,::1] diffQ):
	cdef int N = diffQ.shape[0]
	cdef int K = diffQ.shape[1]
	cdef int i, k
	cdef double sumQ = 0.0
	for i in range(N):
		for k in range(K):
			sumQ += diffQ[i, k]*diffQ[i, k]
	return sumQ

# Update F alpha
cpdef accelUpdateQ(double[:,::1] Q, double[:,::1] Q0, double[:,::1] diff1, \
					double[:,::1] diff3, double alpha, int t):
	cdef int N = Q.shape[0]
	cdef int K = Q.shape[1]
	cdef int i, k
	cdef double sumK
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
cpdef matMinusF(double[:,:,::1] F1, double[:,:,::1] F2, double[:,:,::1] diffF):
	cdef int W = F1.shape[0]
	cdef int K = F1.shape[1]
	cdef int C = F1.shape[2]
	cdef int w, k, c
	for w in range(W):
		for k in range(K):
			for c in range(C):
				diffF[w, k, c] = F1[w, k, c] - F2[w, k, c]

# SumSquare F
cpdef matSumSquareF(double[:,:,::1] diffF):
	cdef int W = diffF.shape[0]
	cdef int K = diffF.shape[1]
	cdef int C = diffF.shape[2]
	cdef int w, k, c
	cdef double sumF = 0.0
	for w in range(W):
		for k in range(K):
			for c in range(C):
				sumF += diffF[w, k, c]*diffF[w, k, c]
	return sumF

# Update F alpha
cpdef accelUpdateF(double[:,:,::1] F, double[:,:,::1] F0, double[:,:,::1] diff1, \
					double[:,:,::1] diff3, double alpha, int t):
	cdef int W = F.shape[0]
	cdef int K = F.shape[1]
	cdef int C = F.shape[2]
	cdef int w, k, c
	cdef double sumY
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

# Root mean squared error (2D)
cpdef float rmse2d(float[:,:] M1, float[:,:] M2):
	cdef int N = M1.shape[0]
	cdef int K = M1.shape[1]
	cdef int i, k
	cdef float res = 0.0
	for i in range(N):
		for k in range(K):
			res = res + (M1[i, k] - M2[i, k])*(M1[i, k] - M2[i, k])
	res = res/(<float>(N*K))
	return sqrt(res)

# Root mean squared error (1D)
cpdef float rmse1d(float[::1] v1, float[::1] v2):
	cdef int N = v1.shape[0]
	cdef int i
	cdef float res = 0.0
	for i in range(N):
		res = res + (v1[i] - v2[i])*(v1[i] - v2[i])
	res = res/<float>N
	return sqrt(res)

##### Cython functions for PCA in HaploNet #####
# Standardize cluster matrix
cpdef standardizeY(signed char[:,::1] L, float[::1] F, float[:,::1] Y, int t):
	cdef int N = L.shape[0]
	cdef int S = L.shape[1]
	cdef int i, s
	with nogil:
		for i in prange(N, num_threads=t):
			for s in range(S):
				Y[i, s] = (L[i, s] - 2*F[s])/sqrt(2*F[s]*(1 - F[s]))

# Covariance estimation
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

# Iterative - Frequency
cpdef emFrequency(float[:,:,::1] L, float[::1] F, float[:,:,::1] H, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int w, i, c
	cdef float sumC, sumN
	with nogil:
		for w in prange(W, num_threads=t):
			for i in range(N):
				sumC = 0.0
				for c in range(C):
					H[w, i, c] = L[w, i, c]*F[w*C + c]
					sumC = sumC + H[w, i, c]
				for c in range(C):
					H[w, i, c] = H[w, i, c]/sumC
			for c in range(C):
				sumN = 0.0
				for i in range(N):
					sumN = sumN + H[w, i, c]
				F[w*C + c] = sumN/<float>N

# Iterative - Center
cpdef generateE(float[:,::1] L, float[::1] F, float[:,::1] H, \
				float[:,::1] Y, int W, int C, int t):
	cdef int N = L.shape[0]
	cdef int i, w, c
	cdef float sumC
	with nogil:
		for i in prange(N, num_threads=t):
			for w in range(W):
				sumC = 0.0
				for c in range(C):
					H[i, w*C + c] = L[i, w*C + c]*F[w*C + c]
					sumC = sumC + H[i, w*C + c]
				for c in range(C):
					H[i, w*C + c] = H[i, w*C + c]/sumC
		for i in prange(0, N, 2, num_threads=t):
			for w in range(W):
				for c in range(C):
					Y[i//2, w*C + c] = H[i+0, w*C + c] + H[i+1, w*C + c] - 2*F[w*C + c]

# Iterative - PCAngsd
cpdef generateP(float[:,::1] L, float[::1] F, float[:,::1] H, \
				float[:,::1] Y, float[:,:] U, float[:] s, float[:,:] V, \
				int W, int C, int t):
	cdef int N = L.shape[0]
	cdef int K = s.shape[0]
	cdef int i, k, w, c
	cdef float sumC
	with nogil:
		for i in prange(N, num_threads=t):
			for w in range(W):
				sumC = 0.0
				for c in range(C):
					Y[i//2, w*C + c] = 0.0
					for k in range(K):
						Y[i//2, w*C + c] += U[i//2, k]*s[k]*V[k, w*C + c]
					Y[i//2, w*C + c] = min(max(Y[i//2, w*C + c] + 2*F[w*C + c], 1e-7), 1-(1e-7))
					H[i, w*C + c] = L[i, w*C + c]*Y[i//2, w*C + c]
					sumC = sumC + H[i, w*C + c]
				for c in range(C):
					H[i, w*C + c] = H[i, w*C + c]/sumC
		for i in prange(0, N, 2, num_threads=t):
			for w in range(W):
				for c in range(C):
					Y[i//2, w*C + c] = H[i+0, w*C + c] + H[i+1, w*C + c] - 2*F[w*C + c]

# Iterative - Standardize
cpdef standardizeE(float[:,::1] Y, float[::1] F, int t):
	cdef int N = Y.shape[0]
	cdef int S = Y.shape[1]
	cdef int i, s
	with nogil:
		for i in prange(N, num_threads=t):
			for s in range(S):
				Y[i, s] = Y[i, s]/sqrt(2*F[s]*(1 - F[s]))
