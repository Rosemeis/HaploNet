"""
HaploNet.
Estimate covariance matrix using neural network likelihoods.
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
parser.add_argument("-t", type=int, default=1,
	help="Number of threads")
parser.add_argument("-out",
	help="Output path/name")
args = parser.parse_args()


##### HaploNet - Covariance matrix #####
print("HaploNet - Covariance matrix estimation")

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

# Containers
Cov = np.zeros((N//2, N//2), dtype=np.float32)

# Convert log-like to like
L -= np.max(L, axis=2, keepdims=True)
L = np.exp(L)
L /= np.sum(L, axis=2, keepdims=True)

# Argmax and estimate haplotype frequencies
L = np.eye(C, dtype=np.int8)[np.argmax(L, axis=2).astype(np.int8)]
F = np.sum(L, axis=1).astype(np.float32).flatten()
F /= float(N)

# Construct data matrix
Y = np.swapaxes(L, 0, 1)
Y = Y.reshape(N, W*C)
Y = Y[0::2] + Y[1::2]

# Filter out low frequency haplotype clusters
mask = F > args.filter
F = F[mask]
Y = np.ascontiguousarray(Y[:, mask])
print("After filtering with threshold (" + str(args.filter) + "): ", Y.shape)

# Estimate covariance matrix
print("Estimating covariance matrix.")
shared_cy.covarianceY(Y, F, Cov, args.t)

# Save covariance matrix
np.savetxt(args.out + ".cov", Cov, fmt="%.7f")
print("Saved covariance matrix as " + args.out + ".cov")
