# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log, exp
from cpython.mem cimport PyMem_Malloc, PyMem_Free

##### Cython functions for FATASH #####
# Safe log sum exp for array
cdef double logsumexp(double* vec, int K):
	cdef double max_v = vec[0]
	cdef double sum_v = 0.0
	for k in range(1, K):
		if vec[k] > max_v:
			max_v = vec[k]
	for k in range(K):
		sum_v += exp(vec[k] - max_v)
	return log(sum_v) + max_v


# Argmax of array
cdef int argmax(double* vec, int K):
	cdef int k, res = 0
	for k in range(1, K):
		if vec[k] > vec[res]:
			res = k
	return res


# Max of array
cdef double maxarr(double* vec, int K):
	cdef int k
	cdef double res = vec[0]
	for k in range(1, K):
		if vec[k] > res:
			res = vec[K]
	return res


# Estimate all emission probabilities
cpdef calcEmission(float[:,:,::1] L, float[:,:,::1] F, float[:,:,::1] E, int t):
	cdef int W = L.shape[0]
	cdef int N = L.shape[1]
	cdef int C = L.shape[2]
	cdef int K = F.shape[1]
	cdef int w, i, c, k
	with nogil:
		for w in prange(W, num_threads=t):
			for i in range(N):
				for k in range(K):
					for c in range(C):
						E[w, i, k] += L[w, i, c]*F[w, k, c]
					E[w, i, k] = log(max(E[w, i, k], 1e-8))


# Create transition matrix
cpdef calcTransition(float[:,::1] T, float[::1] Qi, float a):
	cdef int K = T.shape[0]
	cdef int i, j
	for i in range(K):
		for j in range(K):
			if i == j:
				T[i,j] = log((1.0 - exp(-a))*Qi[i] + exp(-a))
			else:
				T[i,j] = log((1.0 - exp(-a))*Qi[i])

# Create transition matrix with distance
cpdef calcTransitionDist(float[:, :,::1] T, float[::1] Qi, float a, float[::1] W):
	cdef int K = T.shape[1]
	cdef int nW = T.shape[0]
	cdef int i, j, w
	for w in range(nW):
		for i in range(K):
			for j in range(K):
				## TODO check w index is correct for first window. very tired
				if i == j:
					T[w,i,j] = log((1.0 - exp(-a*W[w]))*Qi[i] + exp(-a*W[w]))
				else:
					T[w,i,j] = log((1.0 - exp(-a*W[w]))*Qi[i])

# Log-likelihood function
cpdef double loglike(float[:,:,::1] E, float[::1] Qi, float[:,::1] T, int i):
	cdef int W = E.shape[0]
	cdef int K = E.shape[2]
	cdef int w, k, k1, k2
	cdef double loglike
	cdef double* logP = <double*>PyMem_Malloc(sizeof(double)*K)
	cdef double* logK = <double*>PyMem_Malloc(sizeof(double)*K)
	for k in range(K):
		logP[k] = E[0, i, k] + log(Qi[k])
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				logK[k2] = T[k1,k2] + logP[k2]
			logP[k1] = logsumexp(logK, K) + E[w, i, k1]
	loglike = logsumexp(logP, K)

	# Release memory
	PyMem_Free(logP)
	PyMem_Free(logK)
	return loglike


# Forward backward algorithm
cpdef fwdbwd(float[:,:,::1] E, float[::1] Qi, float[:,:,::1] P, float[:,::1] T, int i):
	cdef int W = E.shape[0]
	cdef int K = E.shape[2]
	cdef int w, k, k1, k2
	cdef double ll_fwd, ll_bwd
	cdef double* res_fwd = <double*>PyMem_Malloc(sizeof(double)*W*K)
	cdef double* res_bwd = <double*>PyMem_Malloc(sizeof(double)*W*K)
	cdef double* tmp_fwd = <double*>PyMem_Malloc(sizeof(double)*K)
	cdef double* tmp_bwd = <double*>PyMem_Malloc(sizeof(double)*K)

	# Forward
	for k in range(K):
		res_fwd[0*K + k] = E[0, i, k] + log(Qi[k])
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				tmp_fwd[k2] = T[k1,k2] + res_fwd[(w-1)*K + k2]
			res_fwd[w*K + k1] = logsumexp(tmp_fwd, K) + E[w, i, k1]

	# Log-likelihood forward
	for k in range(K):
		tmp_fwd[k] = res_fwd[(W-1)*K + k]
	ll_fwd = logsumexp(tmp_fwd, K)

	# Backward
	for k in range(K):
		res_bwd[(W-1)*K + k] = 0.0
	for w in range(W-2, -1, -1):
		for k1 in range(K):
			for k2 in range(K):
				tmp_bwd[k2] = T[k2,k1] + res_bwd[(w+1)*K + k2] + E[w+1, i, k2]
			res_bwd[w*K + k1] = logsumexp(tmp_bwd, K)

	# Log-likelihood backward
	for k in range(K):
		tmp_bwd[k] = res_bwd[0*K + k] + E[0, i, k] + log(Qi[k])
	ll_bwd = logsumexp(tmp_bwd, K)

	# Check if log-likelihoods matches
	assert abs(ll_fwd - ll_bwd)/ll_fwd < 1e-4

	# Update posterior
	for w in range(W):
		for k in range(K):
			P[w, i, k] = res_fwd[w*K + k] + res_bwd[w*K + k] - ll_fwd

	# Release memory
	PyMem_Free(res_fwd)
	PyMem_Free(res_bwd)
	PyMem_Free(tmp_fwd)
	PyMem_Free(tmp_bwd)


# Viterbi algorithm
cpdef viterbi(float[:,:,::1] E, float[::1] Qi, int[:,::1] V, float[:,::1] T, int i):
	cdef int W = E.shape[0]
	cdef int K = E.shape[2]
	cdef int w, k, k1, k2
	cdef double* res_vit = <double*>PyMem_Malloc(sizeof(double)*W*K)
	cdef double* tmp_vit = <double*>PyMem_Malloc(sizeof(double)*K)
	cdef int* arg_vit = <int*>PyMem_Malloc(sizeof(int)*W*K)

	# Viterbi run
	for k in range(K):
		res_vit[0*K + k] = log(Qi[k]) + E[0, i, k]
	for w in range(1, W):
		for k1 in range(K):
			for k2 in range(K):
				tmp_vit[k2] = T[k1,k2] + res_vit[(w-1)*K + k2]
			res_vit[w*K + k1] = maxarr(tmp_vit, K) + E[w, i, k1]
			arg_vit[w*K + k1] = argmax(tmp_vit, K)
	for k in range(K):
		tmp_vit[k] = res_vit[(W-1)*K + k]
	V[W-1, i] = argmax(tmp_vit, K)
	for w in range(W-1, 0, -1):
		V[w-1, i] = arg_vit[w*K + V[w, i]]

	# Free memory
	PyMem_Free(res_vit)
	PyMem_Free(tmp_vit)
	PyMem_Free(arg_vit)
