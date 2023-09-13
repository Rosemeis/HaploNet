"""
HaploNet.
EM algorithm for estimating ancestry proportions and haplotype frequencies.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from haplonet import shared_cy

# Main function
def main(args):
	# Function for EM step
	def emStep(L, F, Q, F_new, Q_new, Q_anc, t):
		shared_cy.emLoop(L, F, Q, F_new, Q_new, t)
		if Q_anc is None:
			shared_cy.updateQ(Q_new, Q)
		else:
			shared_cy.superQ(Q_new, Q, Q_anc)

	# Function for accelerated EM step
	def emStepAccel(L, F, Q, F_new, Q_new, diffF, diffQ, Q_anc, t):
		Fprev = np.copy(F)
		Qprev = np.copy(Q)
		shared_cy.emLoop(L, F, Q, F_new, Q_new, t)
		if Q_anc is None:
			shared_cy.updateQ(Q_new, Q)
		else:
			shared_cy.superQ(Q_new, Q, Q_anc)
		shared_cy.matMinusF(F, Fprev, diffF)
		shared_cy.matMinusQ(Q, Qprev, diffQ)

	# Function for EM step of Q estimation only
	def emTest(L_test, F, Q_test, Q_new, t):
		shared_cy.emQ(L_test, F, Q_test, Q_new, t)
		shared_cy.updateQ(Q_new, Q_test)



	##### HaploNet - EM #####
	print(f"HaploNet - EM algorithm - K={args.K} - seed={args.seed}")
	print("Estimating ancestry proportions and window-based haplotype frequencies.\n")

	# Check input
	assert (args.filelist is not None) or (args.like is not None), \
		"No input data (-f or -l)"
	assert args.K is not None, "Must provide number of ancestral components (-K)!"
	assert not ((args.training is not None) and (args.supervised is not None)), \
		"You can't use --training and --supervised at the same time!"

	# Load data (and concatentate across windows)
	print("Loading log-likelihood file(s).")
	if args.filelist is not None:
		L_list = []
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				L_list.append(np.load(chr.strip("\n")))
				print(f"\rParsed file #{file_c}", end="")
				file_c += 1
			print(".")
		L = np.concatenate(L_list, axis=0)
		del L_list
	else:
		L = np.load(args.like)
	W = L.shape[0]
	N = L.shape[1]//2
	C = L.shape[2]
	print(f"Loaded {2*N} haplotypes, {W} windows, {C} clusters.")

	# Convert log-like to like
	shared_cy.createLikes(L, args.threads)
	if args.training is not None:
		print("Estimating F-matrix only using held-in samples!")
		train = np.loadtxt(args.training, dtype=int).astype(bool)
		assert train.shape[0] == N, "Number samples differ between files!"
		L_test = np.ascontiguousarray(L[:,np.invert(train.repeat(2)),:])
		N_test = L_test.shape[1]//2
		print(f"{N-N_test}/{N} held-in samples.")
		L = np.ascontiguousarray(L[:,train.repeat(2),:])
		N = L.shape[1]//2

	# Initiate containers
	np.random.seed(args.seed) # Set random seed
	Q = np.random.rand(N, args.K)
	Q /= np.sum(Q, axis=1, keepdims=True)
	F = np.random.rand(W, args.K, C)
	F /= np.sum(F, axis=2, keepdims=True)
	Q_new = np.zeros((W, N, args.K))
	F_new = np.zeros((W, args.K, C))
	logVec = np.zeros(W)

	# Semi-supervised setting
	if args.supervised is not None:
		print("Ancestry estimation in supervised mode!")
		Q_anc = np.loadtxt(args.supervised, dtype=np.int32)
		assert Q_anc.shape[0] == Q.shape[0], "Number of samples differ between files!"
		assert np.max(Q_anc) <= args.K, "Wrong supervised ancestry assignment!"
		print(f"{np.sum(Q_anc > 0)} individuals with fixed ancestry.")
		shared_cy.setupQ(Q, Q_anc)
	else:
		Q_anc = None

	# Initialization log-likelihood
	shared_cy.logLike(L, F, Q, logVec, args.threads)
	curLL = np.sum(logVec, dtype=float)
	print(f"(0) Log-likelihood: {np.round(curLL, 1)}")

	# Acceleration containers
	diffF1 = np.zeros((W, args.K, C))
	diffF2 = np.zeros((W, args.K, C))
	diffF3 = np.zeros((W, args.K, C))
	diffQ1 = np.zeros((N, args.K))
	diffQ2 = np.zeros((N, args.K))
	diffQ3 = np.zeros((N, args.K))
	stepMax_F = 4.0
	stepMax_Q = 4.0

	# Run EM SqS3
	for i in range(1, args.iter + 1):
		F0 = np.copy(F)
		Q0 = np.copy(Q)

		# Acceleration step 1
		emStepAccel(L, F, Q, F_new, Q_new, diffF1, diffQ1, Q_anc, args.threads)
		sr2_F = shared_cy.matSumSquareF(diffF1)
		sr2_Q = shared_cy.matSumSquareQ(diffQ1)

		# Acceleration step 2
		emStepAccel(L, F, Q, F_new, Q_new, diffF2, diffQ2, Q_anc, args.threads)
		shared_cy.matMinusF(diffF2, diffF1, diffF3)
		shared_cy.matMinusQ(diffQ2, diffQ1, diffQ3)
		sv2_F = shared_cy.matSumSquareF(diffF3)
		sv2_Q = shared_cy.matSumSquareQ(diffQ3)
		alpha_F = max(1.0, np.sqrt(sr2_F/sv2_F))
		alpha_Q = max(1.0, np.sqrt(sr2_Q/sv2_Q))
		if alpha_F > stepMax_F:
			alpha_F = min(alpha_F, stepMax_F)
			stepMax_F *= 4.0
		if alpha_Q > stepMax_Q:
			alpha_Q = min(alpha_Q, stepMax_Q)
			stepMax_Q *= 4.0

		# Update matrices and map to domain
		shared_cy.accelUpdateF(F, F0, diffF1, diffF3, alpha_F, args.threads)
		if Q_anc is None:
			shared_cy.accelUpdateQ(Q, Q0, diffQ1, diffQ3, alpha_Q)
		else:
			shared_cy.accelSuperQ(Q, Q0, diffQ1, diffQ3, Q_anc, alpha_Q)
		emStep(L, F, Q, F_new, Q_new, Q_anc, args.threads) # Stabilization step
		
		# Convergence check
		if i % args.check == 0:
			shared_cy.logLike(L, F, Q, logVec, args.threads)
			newLL = np.sum(logVec, dtype=float)
			print(f"({i})\tLog-likelihood:\t{np.round(newLL, 1)}", flush=True)
			if abs(newLL - curLL) < args.tole:
				print("EM algorithm converged.")
				print(f"Final log-likelihood:\t{np.round(newLL, 1)}", flush=True)
				break
			if i == args.iter:
				print("EM algorithm did not converge!")
			curLL = newLL

	# Save Q and F
	if args.training is None:
		np.savetxt(f"{args.out}.q", Q, fmt="%.7f")
		print(f"Saved admixture proportions as {args.out}.q")
	np.save(args.out + ".f", F.astype(float))

	# Estimate Q for projected samples using fixed F
	if args.training is not None:
		print("\nEstimating admixture proportions for held-out individuals.")
		Q_test = np.random.rand(N_test, args.K)
		Q_test /= np.sum(Q_test, axis=1, keepdims=True)
		Q_new = np.zeros((W, N_test, args.K))

		# Initialization log-likelihood
		shared_cy.logLike(L_test, F, Q_test, logVec, args.threads)
		curLL = np.sum(logVec, dtype=float)
		print(f"(0)\tLog-likelihood:\t{np.round(curLL, 1)}")

		# Run EM for Q only in held-out individuals
		for i in range(1, args.iter + 1):
			emTest(L_test, F, Q_test, Q_new, args.threads) # Stabilization step

			# Convergence check
			if i % args.check == 0:
				shared_cy.logLike(L_test, F, Q_test, logVec, args.threads)
				newLL = np.sum(logVec, dtype=float)
				print(f"({i})\tLog-likelihood:\t{np.round(newLL, 1)}", flush=True)
				if abs(newLL - curLL) < 0.1:
					print("EM algorithm converged.")
					print(f"Final log-likelihood:\t{np.round(newLL, 1)}", flush=True)
					break
				if i == args.iter:
					print("EM algorithm did not converge!")
				curLL = newLL

		# Save admixture proportions of held-out samples
		Q_final = np.zeros((train.shape[0], args.K))
		Q_final[train,:] = Q
		Q_final[np.invert(train),:] = Q_test
		np.savetxt(f"{args.out}.q", Q_final, fmt="%.7f")
		print(f"Saved admixture proportions as {args.out}.q")



##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet admix'!"
