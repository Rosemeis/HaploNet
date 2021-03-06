"""
HaploNet.
EM algorithm for estimating ancestry proportions and haplotype frequencies.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import argparse
import shared_cy

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-filelist",
	help="Filelist with paths to multiple log-likelihood files")
parser.add_argument("-like",
	help="Path to single log-likelihood file")
parser.add_argument("-K", type=int,
	help="Number of ancestral components")
parser.add_argument("-iter", type=int, default=5000,
	help="Maximum number of iterations")
parser.add_argument("-tole", type=float, default=0.1,
	help="Difference in loglike between args.check iterations")
parser.add_argument("-tole_q", type=float, default=1e-6,
	help="Tolerance for convergence of Q matrix")
parser.add_argument("-no_accel", action="store_true",
	help="Turn off SqS3 acceleration")
parser.add_argument("-threads", type=int, default=1,
	help="Number of threads")
parser.add_argument("-check", type=int, default=50,
	help="Calculating loglike for every i-th iteration")
parser.add_argument("-check_q", action="store_true",
	help="Check convergence for change in Q matrix")
parser.add_argument("-seed", type=int, default=0,
	help="Random seed")
parser.add_argument("-out",
	help="Output path/name")
args = parser.parse_args()

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
print("Loaded data: ", L.shape)
W, N, C = L.shape

# Convert log-like to like
shared_cy.createLikes(L, args.threads)

# Initiate containers
np.random.seed(args.seed) # Set random seed
Q = np.random.rand(N//2, args.K).astype(np.float32)
Q /= np.sum(Q, axis=1, keepdims=True)
F = np.random.rand(W, args.K, C).astype(np.float32)
F /= np.sum(F, axis=2, keepdims=True)
Qnew = np.zeros((W, N//2, args.K), dtype=np.float32)
Fnew = np.zeros((W, args.K, C), dtype=np.float32)
logVec = np.zeros(W, dtype=np.float32)

# Acceleration containers
if not args.no_accel:
	print("Using accelerated EM scheme (SqS3)")
	diffF_1 = np.zeros((W, args.K, C), dtype=np.float32)
	diffF_2 = np.zeros((W, args.K, C), dtype=np.float32)
	diffF_3 = np.zeros((W, args.K, C), dtype=np.float32)
	diffQ_1 = np.zeros((N//2, args.K), dtype=np.float32)
	diffQ_2 = np.zeros((N//2, args.K), dtype=np.float32)
	diffQ_3 = np.zeros((N//2, args.K), dtype=np.float32)

# Run EM
for i in range(args.iter):
	# SqS3 - EM acceleration
	if not args.no_accel:
		if i == 0:
			emStep(L, F, Q, Fnew, Qnew, args.threads)
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

		# Update matrices and map to domain
		shared_cy.accelUpdateF(F, F0, diffF_1, diffF_3, alpha_F, args.threads)
		shared_cy.accelUpdateQ(Q, Q0, diffQ_1, diffQ_3, alpha_Q, args.threads)
	else:
		emStep(L, F, Q, Fnew, Qnew, args.threads)
	if i == 0:
		shared_cy.logLike(L, F, Q, logVec, args.threads)
		curLL = np.sum(logVec, dtype=float)
		print("Iteration " + str(i+1) + ": " + str(curLL))
		if args.check_q:
			Qold = np.copy(Q)
	if (i > 0) and ((i+1) % args.check == 0):
		shared_cy.logLike(L, F, Q, logVec, args.threads)
		newLL = np.sum(logVec, dtype=float)
		print("Iteration " + str(i+1) + ": " + str(newLL))
		if abs(newLL - curLL) < args.tole:
			print("EM algorithm converged.")
			break
		curLL = newLL
		if args.check_q:
			diff = shared_cy.rmse2d(Q, Qold)
			if diff < args.tole_q:
				print("EM algorithm converged (due to change in Q): ", diff)
				break
			Qold = np.copy(Q)

# Save Q and F
np.savetxt(args.out + ".q", Q, fmt="%.7f")
np.save(args.out + ".f", F.astype(float))
print("Saved admixture proportions as " + args.out + ".q")
