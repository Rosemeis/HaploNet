import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt, log

##### Cython functions for EM algorithm in HaploNet #####
# Create monster matrix
@boundscheck(False)
@wraparound(False)
cpdef emLoop(float[:,:,::1] L, float[:,:,::1] F, float[:,::1] Q, \
				float[:,:,::1] Fnew, float[:,:,::1] Qnew, \
				float[:,::1] Psum, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int K = L.shape[2]
	cdef int C = Q.shape[1]
	cdef int w, d, i, a, k, c
	cdef float fStar, postKY, sumY
	with nogil:
		for w in prange(W, num_threads=t):
			# Reset Fnew
			for c in range(C):
				for k in range(K):
					Fnew[w, c, k] = 0.0

			# Reset Qnew
			for d in range(N//2):
				for c in range(C):
					Qnew[w, d, c] = 0.0

			# Run EM
			for i in range(0, N, 2):
				for a in range(2):
					Psum[w, i+a] = 0.0
					for k in range(K):
						fStar = 0.0
						for c in range(C):
							fStar = fStar + F[w, c, k]*Q[i//2, c]
						Psum[w, i+a] = Psum[w, i+a] + L[w, i+a, k]*fStar
					for k in range(K):
						for c in range(C):
							if Psum[w, i+a] > 1e-8:
								postKY = L[w, i+a, k]*F[w, c, k]*Q[i//2, c]/Psum[w, i+a]
								Fnew[w, c, k] += postKY
								Qnew[w, i//2, c] += postKY

			# Update F
			for c in range(C):
				sumY = 0.0
				for k in range(K):
					sumY = sumY + Fnew[w, c, k]
				for k in range(K):
					F[w, c, k] = Fnew[w, c, k]/sumY


# Q update
@boundscheck(False)
@wraparound(False)
cpdef updateQ(float[:,:,::1] Qnew, float[:,::1] Q, int t):
	cdef int W = Qnew.shape[0]
	cdef int N = Qnew.shape[1]
	cdef int C = Qnew.shape[2]
	cdef int w, i, c
	with nogil:
		for i in prange(N, num_threads=t):
			for c in range(C):
				Q[i, c] = 0.0
				for w in range(W):
					Q[i, c] += Qnew[w, i, c]
			for c in range(C):
				Q[i, c] /= 2.0*float(W)


# Log-likelihood
@boundscheck(False)
@wraparound(False)
cpdef logLike(float[:,:,::1] L, float[:,:,::1] F, float[:,::1] Q, \
				float[::1] logVec, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int K = L.shape[2]
	cdef int C = Q.shape[1]
	cdef int w, i, a, c, k
	cdef float fStar, partLike
	with nogil:
		for w in prange(W, num_threads=t):
			logVec[w] = 0.0
			for i in range(0, N, 2):
				for a in range(2):
					partLike = 0.0
					for k in range(K):
						fStar = 0.0
						for c in range(C):
							fStar = fStar + F[w, c, k]*Q[i//2, c]
						partLike = partLike + L[w, i+a, k]*fStar
					logVec[w] += log(partLike)


### Acceleration functions
# Minus Q
@boundscheck(False)
@wraparound(False)
cpdef matMinusQ(float[:,::1] Q1, float[:,::1] Q2, float[:,::1] diffQ):
	cdef int N = Q1.shape[0]
	cdef int C = Q1.shape[1]
	cdef int i, c
	for i in range(N):
		for c in range(C):
			diffQ[i, c] = Q1[i, c] - Q2[i, c]

# SumSquare Q
@boundscheck(False)
@wraparound(False)
cpdef matSumSquareQ(float[:,::1] diffQ):
	cdef int N = diffQ.shape[0]
	cdef int C = diffQ.shape[1]
	cdef int i, c
	cdef float sumQ = 0.0
	for i in range(N):
		for c in range(C):
			sumQ += diffQ[i, c]*diffQ[i, c]
	return sumQ

# Update F alpha
@boundscheck(False)
@wraparound(False)
cpdef accelUpdateQ(float[:,::1] Q, float[:,::1] Q0, float[:,::1] diff1, \
					float[:,::1] diff3, float alpha, int t):
	cdef int N = Q.shape[0]
	cdef int C = Q.shape[1]
	cdef int i, c
	cdef float sumK
	with nogil:
		for i in prange(N, num_threads=t):
			sumK = 0.0
			for c in range(C):
				Q[i, c] = Q0[i, c] + 2*alpha*diff1[i, c] + alpha*alpha*diff3[i, c]
				Q[i, c] = min(max(Q[i, c], 1e-7), 1-(1e-7))
				sumK += Q[i, c]
			for c in range(C):
				Q[i, c] /= sumK

# Minus F
@boundscheck(False)
@wraparound(False)
cpdef matMinusF(float[:,:,::1] F1, float[:,:,::1] F2, float[:,:,::1] diffF):
	cdef int W = F1.shape[0]
	cdef int C = F1.shape[1]
	cdef int K = F1.shape[2]
	cdef int w, c, k
	for w in range(W):
		for c in range(C):
			for k in range(K):
				diffF[w, c, k] = F1[w, c, k] - F2[w, c, k]

# SumSquare F
@boundscheck(False)
@wraparound(False)
cpdef matSumSquareF(float[:,:,::1] diffF):
	cdef int W = diffF.shape[0]
	cdef int C = diffF.shape[1]
	cdef int K = diffF.shape[2]
	cdef int w, c, k
	cdef float sumF = 0.0
	for w in range(W):
		for c in range(C):
			for k in range(K):
				sumF += diffF[w, c, k]*diffF[w, c, k]
	return sumF

# Update F alpha
@boundscheck(False)
@wraparound(False)
cpdef accelUpdateF(float[:,:,::1] F, float[:,:,::1] F0, float[:,:,::1] diff1, \
					float[:,:,::1] diff3, float alpha, int t):
	cdef int W = F.shape[0]
	cdef int C = F.shape[1]
	cdef int K = F.shape[2]
	cdef int w, c, k
	cdef float sumY
	with nogil:
		for w in prange(W, num_threads=t):
			for c in range(C):
				sumY = 0.0
				for k in range(K):
					F[w, c, k] = F0[w, c, k] + 2*alpha*diff1[w, c, k] + alpha*alpha*diff3[w, c, k]
					F[w, c, k] = min(max(F[w, c, k], 1e-7), 1-(1e-7))
					sumY += F[w, c, k]
				for k in range(K):
					F[w, c, k] /= sumY
