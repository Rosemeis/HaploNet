# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
from libc.math cimport sqrt, log, exp

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t
DTYPE2 = np.int16
ctypedef np.int16_t DTYPE2_t
ctypedef vector[unsigned char] char_vec

##### Cython functions for HaploNet #####
# Read VCF/BCF file into NumPy array
cpdef np.ndarray[DTYPE_t, ndim=2] readVCF(v_file, int n):
	cdef:
		int i, j, m
		np.ndarray[DTYPE_t, ndim=2] G_np
		np.ndarray[DTYPE2_t, ndim=2] geno
		char_vec G_var = char_vec(2*n)
		vector[char_vec] G
		unsigned char *G_ptr
	for var in v_file: # Loop through VCF file
		geno = var.genotype.array()
		for i in range(n):
			G_var[2*i+0] = <unsigned char>geno[i,0]
			G_var[2*i+1] = <unsigned char>geno[i,1]
		G.push_back(G_var)
	m = G.size()
	
	# Fill up and return NumPy array
	G_np = np.empty((m, 2*n), dtype=DTYPE)	
	for j in range(m):
		G_ptr = &G[j][0]
		G_np[j] = np.asarray(<unsigned char[:(2*n)]>G_ptr)
	return G_np

# Convert loglikes to normalized likes
cpdef void createLikes(float[:,:,::1] L, int t) nogil:
	cdef:
		int W = L.shape[0]
		int N = L.shape[1]
		int C = L.shape[2]
		int w, i, c
		float tmpMax, tmpSum
	for w in prange(W, num_threads=t):
		for i in range(N):
			tmpMax = L[w,i,0]
			tmpSum = 0.0
			for c in range(1, C):
				if L[w,i,c] > tmpMax:
					tmpMax = L[w,i,c]
			for c in range(C):
				L[w,i,c] = L[w,i,c] - tmpMax
				L[w,i,c] = max(1e-7, exp(L[w,i,c]))
				tmpSum = tmpSum + L[w,i,c]
			for c in range(C):
				L[w,i,c] = L[w,i,c]/tmpSum

# Setup semi-supervised Q
cpdef void setupQ(double[:,::1] Q, int[::1] Q_anc) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
	for i in range(N):
		if Q_anc[i] > 0:
			for k in range(K):
				if (Q_anc[i]-1) == k:
					Q[i,k] = 1-(1e-7)
				else:
					Q[i,k] = 1e-7

# Setup semi-supervised F
cpdef void setupF(float[:,:,::1] L, double[:,:,::1] F, double[:,:,::1] F_new, \
		int[::1] Q_anc, int K) nogil:
	cdef:
		int W = L.shape[0]
		int N = L.shape[1]
		int C = L.shape[2]
		int w, c, k, n
		double sumY
	for w in range(W):
		for k in range(K):
			sumY = 0.0
			for i in range(N):
				n = 0
				if (Q_anc[i//2]-1) == k:
					n = n + 1
					for c in range(C):
						F_new[w,k,c] += L[w,i,c]
			for c in range(C):
				F_new[w,k,c] /= <double>n
				sumY = sumY + F_new[w,k,c]
			for c in range(C):
				F[w,k,c] = F_new[w,k,c]/sumY

# Main EM iteration
cpdef void emLoop(float[:,:,::1] L, double[:,:,::1] F, double[:,::1] Q, \
		double[:,:,::1] F_new, double[:,:,::1] Q_new, int t) nogil:
	cdef:
		int W = L.shape[0]
		int N = L.shape[1]
		int C = L.shape[2]
		int K = Q.shape[1]
		int w, d, i, a, c, k
		double pSum, postKY, sumY
	for w in prange(W, num_threads=t):
		# Reset matrices
		for k in range(K):
			for c in range(C):
				F_new[w,k,c] = 0.0
		for d in range(N//2):
			for k in range(K):
				Q_new[w,d,k] = 0.0

		# Run EM
		for i in range(N):
			pSum = 0.0
			for k in range(K):
				for c in range(C):
					pSum = pSum + L[w,i,c]*F[w,k,c]*Q[i//2,k]
			for k in range(K):
				for c in range(C):
					if pSum > 1e-7:
						postKY = L[w,i,c]*F[w,k,c]*Q[i//2,k]/pSum
						F_new[w,k,c] += postKY
						Q_new[w,i//2,k] += postKY

		# Update F
		for k in range(K):
			sumY = 0.0
			for c in range(C):
				sumY = sumY + F_new[w,k,c]
			for c in range(C):
				F[w,k,c] = F_new[w,k,c]/sumY

# EM loop for Q only (training mode)
cpdef void emQ(float[:,:,::1] L, double[:,:,::1] F, double[:,::1] Q, \
		double[:,:,::1] Q_new, int t) nogil:
	cdef:
		int W = L.shape[0]
		int N = L.shape[1]
		int C = L.shape[2]
		int K = Q.shape[1]
		int w, d, i, a, c, k
		double pSum
	for w in prange(W, num_threads=t):
		# Reset Q_new
		for d in range(N//2):
			for k in range(K):
				Q_new[w,d,k] = 0.0

		# Run EM
		for i in range(N):
			pSum = 0.0
			for k in range(K):
				for c in range(C):
					pSum = pSum + L[w,i,c]*F[w,k,c]*Q[i//2,k]
			for k in range(K):
				for c in range(C):
					if pSum > 1e-7:
						Q_new[w,i//2,k] += L[w,i,c]*F[w,k,c]*Q[i//2,k]/pSum

# Q update
cpdef void updateQ(double[:,:,::1] Q_new, double[:,::1] Q) nogil:
	cdef:
		int W = Q_new.shape[0]
		int N = Q_new.shape[1]
		int K = Q_new.shape[2]
		int w, i, k
		double sumQ
	for i in range(N):
		sumQ = 0.0
		for k in range(K):
			Q[i,k] = 0.0
			for w in range(W):
				Q[i,k] += Q_new[w,i,k]
		for k in range(K):
			Q[i,k] /= 2.0*float(W)
			Q[i,k] = min(max(Q[i,k], 1e-7), 1-(1e-7))
			sumQ = sumQ + Q[i,k]
		for k in range(K):
			Q[i,k] = Q[i,k]/sumQ

# Q update in semi-supervised setting 
cpdef void superQ(double[:,:,::1] Q_new, double[:,::1] Q, int[::1] Q_anc) nogil:
	cdef:
		int W = Q_new.shape[0]
		int N = Q_new.shape[1]
		int K = Q_new.shape[2]
		int w, i, k
		double sumQ
	for i in range(N):
		if Q_anc[i] == 0:
			sumQ = 0.0
			for k in range(K):
				Q[i,k] = 0.0
				for w in range(W):
					Q[i,k] += Q_new[w,i,k]
			for k in range(K):
				Q[i,k] /= 2.0*float(W)
				Q[i,k] = min(max(Q[i,k], 1e-7), 1-(1e-7))
				sumQ = sumQ + Q[i,k]
			for k in range(K):
				Q[i,k] = Q[i,k]/sumQ

# Log-likelihood
cpdef void logLike(float[:,:,::1] L, double[:,:,::1] F, double[:,::1] Q, \
		double[::1] logVec, int t) nogil:
	cdef:
		int W = L.shape[0]
		int N = L.shape[1]
		int C = L.shape[2]
		int K = Q.shape[1]
		int w, i, a, c, k
		double pLike
	for w in prange(W, num_threads=t):
		logVec[w] = 0.0
		for i in range(0, N, 2):
			for a in range(2):
				pLike = 0.0
				for k in range(K):
					for c in range(C):
						pLike = pLike + L[w,i+a,c]*F[w,k,c]*Q[i//2,k]
				logVec[w] += log(pLike)

### Acceleration functions
# Minus Q
cpdef void matMinusQ(double[:,::1] Q1, double[:,::1] Q2, double[:,::1] diffQ) nogil:
	cdef:
		int N = Q1.shape[0]
		int K = Q1.shape[1]
		int i, k
	for i in range(N):
		for k in range(K):
			diffQ[i,k] = Q1[i,k] - Q2[i,k]

# SumSquare Q
cpdef double matSumSquareQ(double[:,::1] diffQ) nogil:
	cdef:
		int N = diffQ.shape[0]
		int K = diffQ.shape[1]
		int i, k
		double sumQ = 0.0
	for i in range(N):
		for k in range(K):
			sumQ = sumQ + diffQ[i,k]*diffQ[i,k]
	return sumQ

# Update Q alpha
cpdef void accelUpdateQ(double[:,::1] Q, double[:,::1] Q0, double[:,::1] diff1, \
		double[:,::1] diff3, double alpha) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumK
	for i in range(N):
		sumK = 0.0
		for k in range(K):
			Q[i,k] = Q0[i,k] + 2*alpha*diff1[i,k] + alpha*alpha*diff3[i,k]
			Q[i,k] = min(max(Q[i,k], 1e-7), 1-(1e-7))
			sumK = sumK + Q[i,k]
		for k in range(K):
			Q[i,k] /= sumK

# Update Q alpha
cpdef void accelSuperQ(double[:,::1] Q, double[:,::1] Q0, double[:,::1] diff1, \
		double[:,::1] diff3, int[::1] Q_anc, double alpha) nogil:
	cdef:
		int N = Q.shape[0]
		int K = Q.shape[1]
		int i, k
		double sumK
	for i in range(N):
		if Q_anc[i] == 0:
			sumK = 0.0
			for k in range(K):
				Q[i,k] = Q0[i,k] + 2*alpha*diff1[i,k] + alpha*alpha*diff3[i,k]
				Q[i,k] = min(max(Q[i,k], 1e-7), 1-(1e-7))
				sumK = sumK + Q[i,k]
			for k in range(K):
				Q[i,k] /= sumK

# Minus F
cpdef void matMinusF(double[:,:,::1] F1, double[:,:,::1] F2, \
		double[:,:,::1] diffF) nogil:
	cdef:
		int W = F1.shape[0]
		int K = F1.shape[1]
		int C = F1.shape[2]
		int w, k, c
	for w in range(W):
		for k in range(K):
			for c in range(C):
				diffF[w,k,c] = F1[w,k,c] - F2[w,k,c]

# SumSquare F
cpdef double matSumSquareF(double[:,:,::1] diffF) nogil:
	cdef:
		int W = diffF.shape[0]
		int K = diffF.shape[1]
		int C = diffF.shape[2]
		int w, k, c
		double sumF = 0.0
	for w in range(W):
		for k in range(K):
			for c in range(C):
				sumF = sumF + diffF[w,k,c]*diffF[w,k,c]
	return sumF

# Update F alpha
cpdef void accelUpdateF(double[:,:,::1] F, double[:,:,::1] F0, double[:,:,::1] diff1, \
		double[:,:,::1] diff3, double alpha, int t) nogil:
	cdef:
		int W = F.shape[0]
		int K = F.shape[1]
		int C = F.shape[2]
		int w, k, c
		double sumY
	for w in prange(W, num_threads=t):
		for k in range(K):
			sumY = 0.0
			for c in range(C):
				F[w,k,c] = F0[w,k,c] + 2*alpha*diff1[w,k,c] + alpha*alpha*diff3[w,k,c]
				F[w,k,c] = min(max(F[w,k,c], 1e-7), 1-(1e-7))
				sumY = sumY + F[w,k,c]
			for c in range(C):
				F[w,k,c] /= sumY

# Root mean squared error (2D)
cpdef float rmse2d(float[:,:] M1, float[:,:] M2) nogil:
	cdef:
		int N = M1.shape[0]
		int K = M1.shape[1]
		int i, k
		float res = 0.0
	for i in range(N):
		for k in range(K):
			res = res + (M1[i,k] - M2[i,k])*(M1[i,k] - M2[i,k])
	res = res/(<float>(N*K))
	return sqrt(res)

# Root mean squared error (1D)
cpdef float rmse1d(float[::1] v1, float[::1] v2) nogil:
	cdef:
		int N = v1.shape[0]
		int i
		float res = 0.0
	for i in range(N):
		res = res + (v1[i] - v2[i])*(v1[i] - v2[i])
	res = res/<float>N
	return sqrt(res)


##### Cython functions for PCA in HaploNet #####
# Standardize cluster matrix
cpdef void standardizeY(signed char[:,::1] L, float[::1] F, float[:,::1] Y, int t) nogil:
	cdef:
		int N = L.shape[0]
		int S = L.shape[1]
		int i, s
	for i in prange(N, num_threads=t):
		for s in range(S):
			Y[i,s] = (L[i,s] - 2*F[s])/sqrt(2*F[s]*(1 - F[s]))

# Covariance estimation
cpdef void covarianceY(signed char[:,::1] L, float[::1] F, float[:,::1] C, int t) nogil:
	cdef:
		int N = L.shape[0]
		int S = L.shape[1]
		int i, j, s
	for i in prange(N, num_threads=t):
		for j in range(i, N):
			for s in range(S):
				C[i,j] += (L[i,s] - 2*F[s])*(L[j,s] - 2*F[s])/(2*F[s]*(1 - F[s]))
			C[i,j] /= <float>S
			C[j,i] = C[i,j]

# Iterative - Frequency
cpdef void emFrequency(float[:,:,::1] L, float[::1] F, float[:,:,::1] H, int t) nogil:
	cdef:
		int W = L.shape[0]
		int N = L.shape[1]
		int C = L.shape[2]
		int w, i, c
		float sumC, sumN
	for w in prange(W, num_threads=t):
		for i in range(N):
			sumC = 0.0
			for c in range(C):
				H[w,i,c] = L[w,i,c]*F[w*C + c]
				sumC = sumC + H[w,i,c]
			for c in range(C):
				H[w,i,c] = H[w,i,c]/sumC
		for c in range(C):
			sumN = 0.0
			for i in range(N):
				sumN = sumN + H[w,i,c]
			F[w*C + c] = sumN/<float>N

# Iterative - Center
cpdef void generateE(float[:,::1] L, float[::1] F, float[:,::1] H, float[:,::1] Y, \
		signed char[::1] mask, int W, int C, int t) nogil:
	cdef:
		int N = L.shape[0]
		int i, w, c, s
		float sumC
	for i in prange(N, num_threads=t):
		for w in range(W):
			sumC = 0.0
			for c in range(C):
				if mask[w*C + c] == 1:
					H[i,w*C + c] = L[i,w*C + c]*F[w*C + c]
					sumC = sumC + H[i,w*C + c]
			for c in range(C):
				if mask[w*C + c] == 1:
					H[i,w*C + c] = H[i,w*C + c]/sumC
	for i in prange(0, N, 2, num_threads=t):
		s = 0
		for w in range(W):
			for c in range(C):
				if mask[w*C + c] == 1:
					Y[i//2,s] = H[i+0,w*C + c] + H[i+1,w*C + c] - 2*F[w*C + c]
					s = s + 1

# Iterative - PCAngsd
cpdef void generateP(float[:,::1] L, float[::1] F, float[:,::1] H, float[:,::1] Y, \
		float[:,:] U, float[:] s, float[:,:] V, signed char[::1] mask, int W, int C, \
		int t) nogil:
	cdef:
		int N = L.shape[0]
		int K = s.shape[0]
		int i, k, w, c, h, s1, s2
		float sumC, sumK
	for i in prange(0, N, 2, num_threads=t):
		s1 = 0
		for w in range(W):
			s2 = 0
			sumK = 0.0
			for c in range(C):
				if mask[w*C + c] == 1:
					Y[i//2,s1+s2] = 0.0
					for k in range(K):
						Y[i//2,s1+s2] += U[i//2,k]*s[k]*V[k,s1+s2]
					Y[i//2,s1+s2] = (Y[i//2,s1+s2] + 2*F[w*C + c])/2.0
					Y[i//2,s1+s2] = min(max(1e-7, Y[i//2,s1+s2]), 1-(1e-7))
					sumK = sumK + Y[i//2,s1+s2]
					s2 = s2 + 1
			for h in range(2):
				s2 = 0
				sumC = 0.0
				for c in range(C):
					if mask[w*C + c] == 1:
						H[i+h,w*C + c] = L[i+h,w*C + c]*(Y[i//2,s1+s2]/sumK)
						sumC = sumC + H[i+h,w*C + c]
						s2 = s2 + 1
				for c in range(C):
					if mask[w*C + c] == 1:
						H[i+h,w*C + c] = H[i+h,w*C + c]/sumC
			s1 = s1 + s2
	for i in prange(0, N, 2, num_threads=t):
		s1 = 0
		for w in range(W):
			for c in range(C):
				if mask[w*C + c] == 1:
					Y[i//2,s1] = H[i+0,w*C + c] + H[i+1,w*C + c] - 2*F[w*C + c]
					s1 = s1 + 1

# Iterative - Standardize
cpdef void standardizeE(float[:,::1] Y, float[::1] F, int t) nogil:
	cdef:
		int N = Y.shape[0]
		int S = Y.shape[1]
		int i, s
	for i in prange(N, num_threads=t):
		for s in range(S):
			Y[i,s] = Y[i,s]/sqrt(2*F[s]*(1 - F[s]))
