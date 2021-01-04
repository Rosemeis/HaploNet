"""
HaploNet.
EM algorithm for estimating ancestry proportions and haplotype frequencies.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import argparse
import admixNN_cy

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-folder",
	help="Path to chromosome folders")
parser.add_argument("-like",
	help="Filename of log-likelihood files")
parser.add_argument("-K", type=int,
	help="Number of ancestral components")
parser.add_argument("-iter", type=int, default=2000,
	help="Maximum number of iterations")
parser.add_argument("-tole", type=float, default=0.1,
	help="Difference in loglike between args.check iterations")
parser.add_argument("-no_accel", action="store_true",
	help="Turn off SqS3 acceleration")
parser.add_argument("-t", type=int, default=1,
	help="Number of threads")
parser.add_argument("-check", type=int, default=50,
	help="Calculating loglike for the args.check operation")
parser.add_argument("-seed", type=int, default=0,
	help="Random seed")
parser.add_argument("-out",
	help="Output path/name")
args = parser.parse_args()

# Function for EM step
def emStep(L, F, Q, Fnew, Qnew, Psum, t):
	admixNN_cy.emLoop(L, F, Q, Fnew, Qnew, Psum, t)
	admixNN_cy.updateQ(Qnew, Q, t)

# Function for EM step
def emStepAccel(L, F, Q, Fnew, Qnew, Psum, diffF, diffQ, t):
	Fprev = np.copy(F)
	Qprev = np.copy(Q)
	admixNN_cy.emLoop(L, F, Q, Fnew, Qnew, Psum, t)
	admixNN_cy.updateQ(Qnew, Q, t)
	admixNN_cy.matMinusF(F, Fprev, diffF)
	admixNN_cy.matMinusQ(Q, Qprev, diffQ)


##### HaploNet - EM #####
print("HaploNet - EM algorithm")
print("Estimating ancestry proportions and window-based haplotype frequencies.")

# Load data (and concatentate across windows)
if args.folder is not None:
	L = np.load(args.folder + "/chr1/" + args.like)
	for i in range(2, 23):
		L = np.concatenate((L, np.load(args.folder + "/chr" + str(i) + "/" + \
							args.like)), axis=0)
else:
	L = np.load(args.like)
print("Loaded data: ", L.shape)
W, N, C = L.shape

# Initiate containers
np.random.seed(args.seed) # Set random seed
Q = np.random.rand(N//2, args.K).astype(np.float32)
Q /= np.sum(Q, axis=1, keepdims=True)
F = np.random.rand(W, args.K, C).astype(np.float32)
F /= np.sum(F, axis=2, keepdims=True)
P = np.zeros((W, N, args.K, C), dtype=np.float32)
Psum = np.zeros((W, N), dtype=np.float32)
Qnew = np.zeros((W, N//2, args.K), dtype=np.float32)
Fnew = np.zeros((W, args.K, C), dtype=np.float32)
logVec = np.zeros(W, dtype=np.float32)

# Convert log-like to like
L -= np.max(L, axis=2, keepdims=True)
L = np.exp(L)
L /= np.sum(L, axis=2, keepdims=True)
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
			emStep(L, F, Q, Fnew, Qnew, Psum, args.t)
		F0 = np.copy(F)
		Q0 = np.copy(Q)

		# Acceleration step 1
		emStepAccel(L, F, Q, Fnew, Qnew, Psum, diffF_1, diffQ_1, args.t)
		sr2_F = admixNN_cy.matSumSquareF(diffF_1)
		sr2_Q = admixNN_cy.matSumSquareQ(diffQ_1)

		# Acceleration step 2
		emStepAccel(L, F, Q, Fnew, Qnew, Psum, diffF_2, diffQ_2, args.t)
		admixNN_cy.matMinusF(diffF_2, diffF_1, diffF_3)
		admixNN_cy.matMinusQ(diffQ_2, diffQ_1, diffQ_3)
		sv2_F = admixNN_cy.matSumSquareF(diffF_3)
		sv2_Q = admixNN_cy.matSumSquareQ(diffQ_3)
		alpha_F = max(1.0, np.sqrt(sr2_F/sv2_F))
		alpha_Q = max(1.0, np.sqrt(sr2_Q/sv2_Q))

		# Update matrices and map to domain
		admixNN_cy.accelUpdateF(F, F0, diffF_1, diffF_3, alpha_F, args.t)
		admixNN_cy.accelUpdateQ(Q, Q0, diffQ_1, diffQ_3, alpha_Q, args.t)
	else:
		emStep(L, F, Q, Fnew, Qnew, Psum, args.t)
	if i == 0:
		admixNN_cy.logLike(L, F, Q, logVec, args.t)
		curLL = np.sum(logVec)
		print("Iteration " + str(i+1) + ": " + str(curLL))
	if (i+1) % args.check == 0:
		admixNN_cy.logLike(L, F, Q, logVec, args.t)
		newLL = np.sum(logVec)
		print("Iteration " + str(i+1) + ": " + str(newLL))
		if abs(newLL - curLL) < args.tole:
			print("EM algorithm converged.")
			break
		curLL = newLL

# Save Q and F
np.save(args.out + ".q", Q.astype(float))
np.save(args.out + ".f", F.astype(float))
