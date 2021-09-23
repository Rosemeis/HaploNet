"""
HaploNet.
Perform PCA using neural network likelihoods.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import numpy as np
from haplonet import shared_cy

# Main function
def main(args):
	##### HaploNet - PCA #####
	print("HaploNet - PCA")

	# Check input
	assert (args.filelist is not None) or (args.like is not None), \
			"No input data (-f or -l)"

	# Load data (and concatentate across windows)
	if args.filelist is not None:
		L_list = []
		with open(args.filelist) as f:
			for chr in f:
				L_list.append(np.load(chr.strip("\n")))
		L = np.concatenate(L_list, axis=0)
		del L_list
	else:
		L = np.load(args.like)
	print("Loaded data.", L.shape)
	W, N, C = L.shape

	# Convert log-like to like
	shared_cy.createLikes(L, args.threads)

	# Argmax and estimate haplotype frequencies
	L = np.eye(C, dtype=np.int8)[np.argmax(L, axis=2).astype(np.int8)]
	F = np.sum(L, axis=1).astype(np.float32).flatten()
	F /= float(N)
	if args.freqs:
		np.save(args.out + ".haplotype.freqs", F)

	# Construct data matrix
	L = np.swapaxes(L, 0, 1)
	L = L.reshape(N, W*C)
	if args.unphased:
		print("Assuming unphased genotype data.")
	else:
		L = L[0::2] + L[1::2]
		N = N//2

	# Filter out low frequency haplotype clusters
	mask = (F >= args.filter) & (F <= (1.0 - args.filter))
	F = F[mask]
	L = np.ascontiguousarray(L[:, mask])
	print("After filtering with threshold (" + str(args.filter) + "): ", L.shape)

	# Covariance mode
	if args.cov:
		Cov = np.zeros((N, N), dtype=np.float32)

		# Estimate covariance matrix
		print("Estimating covariance matrix.")
		if args.unphased:
			shared_cy.covarianceY_unphased(L, F, Cov, args.threads)
		else:
			shared_cy.covarianceY(L, F, Cov, args.threads)

		# Save covariance matrix
		np.savetxt(args.out + ".cov", Cov, fmt="%.7f")
		print("Saved covariance matrix as " + args.out + ".cov")
	else:
		from scipy.sparse.linalg import svds
		Y = np.zeros(L.shape, dtype=np.float32)
		if args.unphased:
			shared_cy.standardizeY_unphased(L, F, Y, args.threads)
		else:
			shared_cy.standardizeY(L, F, Y, args.threads)

		# Perform SVD
		print("Performing truncated SVD, extracting " + str(args.n_eig) + \
				" eigenvectors.")
		U, s, V = svds(Y, k=args.n_eig)

		# Save matrices
		np.savetxt(args.out + ".eigenvecs", U[:, ::-1], fmt="%.7f")
		print("Saved eigenvectors as " + args.out + ".eigenvecs.")
		np.savetxt(args.out + ".eigenvals", s[::-1]**2/float(Y.shape[1]), fmt="%.7f")
		print("Saved eigenvalues as " + args.out + ".eigenvals.")
		if args.loadings:
			np.savetxt(args.out + ".loadings", V[::-1,:].T, fmt="%.7f")
			print("Saved loadings as " + args.out + ".loadings.")


##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet pca'!"
