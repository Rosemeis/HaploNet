"""
HaploNet.
Perform PCA using neural network likelihoods.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from scipy.sparse.linalg import svds
from haplonet import shared_cy

# EM - Estimate haplotype cluster frequencies
def estimateF(L, threads):
	W, N, C = L.shape
	F = np.zeros(W*C, dtype=np.float32)
	F.fill(1.0/float(C))
	F_prev = np.copy(F)
	H = np.zeros((W, N, C), dtype=np.float32) # Help container
	for i in range(100):
		shared_cy.emFrequency(L, F, H, threads)
		diff = shared_cy.rmse1d(F, F_prev)
		if diff < 1e-4:
			print("EM (Haplotype frequencies) has converged ({}).".format(i+1))
			break
		F_prev = np.copy(F)
	del H
	return F

# Main function
def main(args):
	##### HaploNet - PCA #####
	print("HaploNet - PCA")

	# Check input
	assert (args.filelist is not None) or (args.like is not None), \
			"No input data (-f or -l)"
	if args.iterative is not None:
		assert not args.cov, "Probabilistic approach don't support cov estimation!"

	# Load data (and concatentate across windows)
	print("Loading log-likelihood file(s).")
	if args.filelist is not None:
		L_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				L_list.append(np.load(chr.strip("\n")))
				print("\rParsed file #" + str(file_c), end="")
				file_c += 1
			print(".")
		L = np.concatenate(L_list, axis=0)
		del L_list
	else:
		L = np.load(args.like)
	print("Loaded {} haplotypes, {} windows, {} clusters.".format(L.shape[1], L.shape[0], L.shape[2]))
	W, N, C = L.shape

	# Convert log-like to like
	shared_cy.createLikes(L, args.threads)
	if args.iterative is not None:
		# Estimate haplotype cluster frequencies
		F = estimateF(L, args.threads)
		if args.freqs:
			np.save(args.out + ".haplotype.freqs", F)

		# Filter out low frequency haplotype clusters
		mask = (F >= args.filter) & (F <= (1.0 - args.filter))
		mask_vec = mask.astype(np.int8)
		T = np.sum(mask, dtype=int) # Number of haplotype clusters to use in optimization

		# Prepare likes
		L = np.swapaxes(L, 0, 1)
		L = np.ascontiguousarray(L.reshape(N, W*C))
		H = np.zeros(L.shape, dtype=np.float32) # Help container
		Y = np.zeros((N//2, T), dtype=np.float32) # Pi container

		# Generate E
		shared_cy.generateE(L, F, H, Y, mask_vec, W, C, args.threads)
		if args.iterative is not None:
			print("Iterative estimation of haplotype cluster frequencies.")
			for i in range(args.iterations):
				U, s, V = svds(Y, k=args.iterative)
				shared_cy.generateP(L, F, H, Y, U, s, V, mask_vec, W, C, args.threads)
				if i > 0:
					diff = shared_cy.rmse2d(Y, Y_prev)
					print("({}) Diff: {}".format(i, np.round(diff, 12)), flush=True)
					if diff < 5e-5:
						print("Iterative estimations have converged.")
						break
				Y_prev = np.copy(Y)
			del U, s, V, Y_prev
		del L, H

		# Standardize dosage matrix
		F = F[mask]
		np.clip(F, 1e-6, (1.0 - 1e-6)) # Ensure non-zero values
		shared_cy.standardizeE(Y, F, args.threads)
	else:
		# Argmax approach
		L = np.eye(C, dtype=np.int8)[np.argmax(L, axis=2).astype(np.int8)]
		F = np.sum(L, axis=1).astype(np.float32).flatten()
		F /= float(N)
		if args.freqs:
			np.save(args.out + ".haplotype.freqs", F)

		# Construct data matrix
		L = np.swapaxes(L, 0, 1)
		L = L.reshape(N, W*C)
		L = L[0::2] + L[1::2]

		# Filter out low frequency haplotype clusters
		mask = (F >= args.filter) & (F <= (1.0 - args.filter))
		F = F[mask]
		np.clip(F, 1e-6, (1.0 - 1e-6)) # Ensure non-zero values
		L = np.ascontiguousarray(L[:, mask])

	# Covariance mode
	if args.cov:
		Cov = np.zeros((N//2, N//2), dtype=np.float32)

		# Estimate covariance matrix
		print("Estimating covariance matrix.")
		shared_cy.covarianceY(L, F, Cov, args.threads)

		# Save covariance matrix
		np.savetxt(args.out + ".cov", Cov, fmt="%.7f")
		print("Saved covariance matrix as " + args.out + ".cov")
	else:
		if args.iterative is None:
			Y = np.zeros(L.shape, dtype=np.float32)
			shared_cy.standardizeY(L, F, Y, args.threads)

		# Perform SVD
		print("Performing truncated SVD, extracting " + str(args.n_eig) + \
				" eigenvectors.")
		U, s, V = svds(Y, k=args.n_eig)

		# Save matrices
		np.savetxt(args.out + ".eigenvecs", U[:, ::-1], fmt="%.7f")
		print("Saved eigenvectors as " + args.out + ".eigenvecs")
		np.savetxt(args.out + ".eigenvals", s[::-1]**2/float(Y.shape[1]), fmt="%.7f")
		print("Saved eigenvalues as " + args.out + ".eigenvals")
		if args.loadings:
			np.savetxt(args.out + ".loadings", V[::-1,:].T, fmt="%.7f")
			print("Saved loadings as " + args.out + ".loadings")


##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet pca'!"
