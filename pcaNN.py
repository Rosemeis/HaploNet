"""
HaploNet.
Perform PCA using neural network likelihoods.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import numpy as np
import shared_cy

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-folder",
	help="Path to chromosome folders")
parser.add_argument("-like",
	help="Filename of log-likelihood files")
parser.add_argument("-filter", type=float, default=0.01,
	help="Threshold for haplotype cluster frequency")
parser.add_argument("-n_chr", type=int, default=22,
	help="Number of chromosomes/scaffolds")
parser.add_argument("-e", type=int, default=10,
	help="Number of eigenvectors to extract")
parser.add_argument("-cov", action="store_true",
	help="Estimate covariance matrix instead of SVD")
parser.add_argument("-threads", type=int, default=1,
	help="Number of threads")
parser.add_argument("-out",
	help="Output path/name")
args = parser.parse_args()


##### HaploNet - PCA #####
print("HaploNet - PCA")

# Load data (and concatentate across windows)
if args.folder is not None:
	L = np.load(args.folder + "/chr1/" + args.like)
	for i in range(2, args.n_chr + 1):
		L = np.concatenate((L, np.load(args.folder + "/chr" + str(i) + "/" + \
							args.like)), axis=0)
else:
	L = np.load(args.like)
print("Loaded data.")
W, N, C = L.shape

# Convert log-like to like
L -= np.max(L, axis=2, keepdims=True)
L = np.exp(L)
L /= np.sum(L, axis=2, keepdims=True)

# Argmax and estimate haplotype frequencies
L = np.eye(C, dtype=np.int8)[np.argmax(L, axis=2).astype(np.int8)]
F = np.sum(L, axis=1).astype(np.float32).flatten()
F /= float(N)

# Construct data matrix
L = np.swapaxes(L, 0, 1)
L = L.reshape(N, W*C)
L = L[0::2] + L[1::2]

# Filter out low frequency haplotype clusters
mask = F > args.filter
F = F[mask]
L = np.ascontiguousarray(L[:, mask])
print("After filtering with threshold (" + str(args.filter) + "): ", L.shape)

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
	from scipy.sparse.linalg import svds
	Y = np.empty(L.shape, dtype=np.float32)
	shared_cy.standardizeY(L, F, Y, args.threads)

	# Perform SVD
	print("Performing truncated SVD, extracting " + str(args.e) + \
			" eigenvectors.")
	U, s, V = svds(Y, k=args.e)

	# Save matrices
	np.savetxt(args.out + ".eigenvecs", U, fmt="%.7f")
	print("Saved eigenvectors as " + args.out + ".eigenvecs.")
	np.savetxt(args.out + ".eigenvals", s**2/float(Y.shape[1]), fmt="%.7f")
	print("Saved eigenvalues as " + args.out + ".eigenvals.")
