"""
HaploNet.
EM algorithm for estimating ancestry proportions and haplotype frequencies.
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import numpy as np
from haplonet import shared_cy

# Main function
def main(args):
	# Function for EM step
	def emStep(L, F, Q, Fnew, Qnew, t):
		shared_cy.emLoop(L, F, Q, Fnew, Qnew, t)
		shared_cy.updateQ(Qnew, Q, t)

	# Function for EM step
	def emStepAccel(L, F, Q, Fnew, Qnew, diffF, diffQ, t):
		Fprev = np.copy(F)
		Qprev = np.copy(Q)
		shared_cy.emLoop(L, F, Q, Fnew, Qnew, t)
		shared_cy.updateQ(Qnew, Q, t)
		shared_cy.matMinusF(F, Fprev, diffF)
		shared_cy.matMinusQ(Q, Qprev, diffQ)


	##### HaploNet - EM #####
	print("HaploNet - EM algorithm - K=" + str(args.K))
	print("Estimating ancestry proportions and window-based haplotype frequencies.")

	# Check input
	assert (args.filelist is not None) or (args.like is not None), \
			"No input data (-f or -l)"
	assert args.K is not None, "Must provide number of ancestral components (-K)!"

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

	# Initiate containers
	np.random.seed(args.seed) # Set random seed
	Q = np.random.rand(N//2, args.K).astype(np.float64)
	Q /= np.sum(Q, axis=1, keepdims=True)
	F = np.random.rand(W, args.K, C).astype(np.float64)
	F /= np.sum(F, axis=2, keepdims=True)
	Qnew = np.zeros((W, N//2, args.K), dtype=np.float64)
	Fnew = np.zeros((W, args.K, C), dtype=np.float64)
	logVec = np.zeros(W, dtype=np.float64)

	# Initialization log-likelihood
	shared_cy.logLike(L, F, Q, logVec, args.threads)
	curLL = np.sum(logVec, dtype=float)
	print("(0) Log-likelihood: {}".format(np.round(curLL, 5)))

	# Acceleration containers
	if not args.no_accel:
		print("Using accelerated EM scheme (SqS3).")
		diffF_1 = np.zeros((W, args.K, C), dtype=np.float64)
		diffF_2 = np.zeros((W, args.K, C), dtype=np.float64)
		diffF_3 = np.zeros((W, args.K, C), dtype=np.float64)
		diffQ_1 = np.zeros((N//2, args.K), dtype=np.float64)
		diffQ_2 = np.zeros((N//2, args.K), dtype=np.float64)
		diffQ_3 = np.zeros((N//2, args.K), dtype=np.float64)
		stepMax_F = 4.0
		stepMax_Q = 4.0

	# Run EM
	for i in range(1, args.iter + 1):
		# SqS3 - EM acceleration
		if not args.no_accel:
			F0 = np.copy(F)
			Q0 = np.copy(Q)

			# Acceleration step 1
			emStepAccel(L, F, Q, Fnew, Qnew, diffF_1, diffQ_1, args.threads)
			sr2_F = shared_cy.matSumSquareF(diffF_1)
			sr2_Q = shared_cy.matSumSquareQ(diffQ_1)

			# Acceleration step 2
			emStepAccel(L, F, Q, Fnew, Qnew, diffF_2, diffQ_2, args.threads)
			shared_cy.matMinusF(diffF_2, diffF_1, diffF_3)
			shared_cy.matMinusQ(diffQ_2, diffQ_1, diffQ_3)
			sv2_F = shared_cy.matSumSquareF(diffF_3)
			sv2_Q = shared_cy.matSumSquareQ(diffQ_3)
			alpha_F = max(1.0, np.sqrt(sr2_F/sv2_F))
			alpha_Q = max(1.0, np.sqrt(sr2_Q/sv2_Q))
			if alpha_F > stepMax_F:
				alpha_F = min(alpha_F, stepMax_F)
				stepMax_F *= 4.0
			if alpha_Q > stepMax_Q:
				alpha_Q = min(alpha_Q, stepMax_Q)
				stepMax_Q *= 4.0

			# Update matrices and map to domain
			shared_cy.accelUpdateF(F, F0, diffF_1, diffF_3, alpha_F, args.threads)
			shared_cy.accelUpdateQ(Q, Q0, diffQ_1, diffQ_3, alpha_Q, args.threads)
		else:
			emStep(L, F, Q, Fnew, Qnew, args.threads)
		if i % args.check == 0:
			shared_cy.logLike(L, F, Q, logVec, args.threads)
			newLL = np.sum(logVec, dtype=float)
			print("({}) Log-likelihood: {}".format(i, np.round(newLL, 5)), flush=True)
			if abs(newLL - curLL) < args.tole:
				print("EM algorithm converged.")
				break
			curLL = newLL

	# Save Q and F
	np.savetxt(args.out + ".q", Q, fmt="%.7f")
	np.save(args.out + ".f", F.astype(float))
	print("Saved admixture proportions as " + args.out + ".q")


##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet admix'!"
