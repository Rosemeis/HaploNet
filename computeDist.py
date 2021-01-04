"""
HaploNet.
Estimate covariance/distance matrix from latent space.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import argparse
import computeDist_cy

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-folder",
	help="Path to chromosome folders")
parser.add_argument("-latent",
	help="Prefix of latent files")
parser.add_argument("-e", action="store_true",
	help="Estimate Euclidean distance")
parser.add_argument("-m", action="store_true",
	help="Estimate Mahalanobis distance")
parser.add_argument("-b", action="store_true",
	help="Estimate Bhattacharyya distance")
parser.add_argument("-manhattan", action="store_true",
	help="Use haplotype cluster assignments")
parser.add_argument("-out",
	help="Output path/name")
parser.add_argument("-t", type=int, default=1,
	help="Number of threads")
args = parser.parse_args()

##### HaploNet - Covar/Dist #####
print("HaploNet - Covariance/Distance estimation")

# Check parsing
assert args.latent is not None, "Must provide prefix to latent file(s)!"
if args.e or args.m or args.b:
	dist = True
else:
	dist = False

# Load data (and concatentate across windows)
if args.folder is not None:
	if args.manhattan:
		Y = np.load(args.folder + "/chr1/" + args.latent + ".y.npy")
	else:
		Z = np.load(args.folder + "/chr1/" + args.latent + ".z.npy")
		if args.m or args.b:
			V = np.load(args.folder + "/chr1/" + args.latent + ".v.npy")
	for i in range(2, 23):
		if args.manhattan:
			Y = np.concatenate((Y, np.load(args.folder + "/chr" + str(i) + \
								"/" + args.latent + ".y.npy")), axis=1)
		else:
			Z = np.concatenate((Z, np.load(args.folder + "/chr" + str(i) + \
								"/" + args.latent + ".z.npy")), axis=1)
			if args.m or args.b:
				V = np.concatenate((V, np.load(args.folder + "/chr" + str(i) + \
									"/" + args.latent + ".v.npy")), axis=1)
else:
	if args.manhattan:
		Y = np.load(args.latent + ".y.npy")
	else:
		Z = np.load(args.latent + ".z.npy")
		if args.m or args.b:
	        	V = np.load(args.latent + ".v.npy")

print("Loaded data")
if args.manhattan:
	N, M, K = Y.shape
	print("Y-shape", Y.shape)
	if args.manhattan:
		D4 = np.zeros((N//2, N//2), dtype=np.float32)
else:
	print("Z-shape", Z.shape)
	N, M, K = Z.shape
	if dist:
		if args.e:
			D1 = np.zeros((N//2, N//2), dtype=np.float32)
		if args.m or args.b:
			eV = np.exp(V)
			if args.m:
				D2 = np.zeros((N//2, N//2), dtype=np.float32)
			if args.b:
				D3 = np.zeros((N//2, N//2), dtype=np.float32)
	else:
		C = np.zeros((N//2, N//2), dtype=np.float32)
		F = np.mean(Z, axis=0)

if args.manhattan:
	if args.manhattan:
		print("Manhattan distance")
		computeDist_cy.ManhattanMatrix(Y, D4, args.t)
else:
	if dist:
		if args.e:
			print("Euclidean distance")
			computeDist_cy.EuclideanMatrix(Z, D1, args.t)
		if args.m:
			print("Mahalanobis distance")
			computeDist_cy.MahalanobisMatrix(Z, eV, D2, args.t)
		if args.b:
			print("Bhattacharyya distance")
			computeDist_cy.BhattacharyyaMatrix(Z, V, eV, D3, args.t)
	else:
		print("Covariance matrix")
		computeDist_cy.CovarianceCluster(Z, F, C, args.t)

# Save matrices
if args.manhattan:
	if args.manhattan:
		np.save(args.out + ".dist.manhattan", D4.astype(float))
else:
	if dist:
		if args.e:
			np.save(args.out + ".dist.euclidean", D1.astype(float))
		if args.m:
			np.save(args.out + ".dist.mahalanobis", D2.astype(float))
		if args.b:
			np.save(args.out + ".dist.bhattacharyya", D3.astype(float))
	else:
		np.save(args.out + ".cov", C.astype(float))
