"""
HaploNet.
HMM for inferring local ancestry tracts.
"""

__author__ = "Jonas Meisner and Kristian Hanghoej"

# Libraries
import argparse
import numpy as np
import scipy.optimize as optim
from haplonet import lahmm_cy
from haplonet import shared_cy

# Log-likehood wrapper for SciPy
def loglike_wrapper(param, *args):
	E, Qi, T, i = args
	lahmm_cy.calcTransition(T, Qi, param)
	return lahmm_cy.loglike(E, Qi, T, i)


# Main function
def main(args):
	##### HaploNet - FATASH #####
	print("HaploNet - FATASH")
	print("Discrete HMM for inferring local ancestry tracts.\n")

	# Check input
	assert (args.filelist is not None) or (args.like is not None), \
			"No log-likelihoods provided (-f or -l)"
	assert args.prop is not None, "No ancestry proportions provided (-q)"
	assert args.freq is not None, "No haplotype cluster frequencies provided (-p)"

	# Load data
	print("Loading data.")
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
	Q = np.genfromtxt(args.prop).astype(np.float32) # Ancestry proportions
	F = np.load(args.freq).astype(np.float32) # Ancestral haplotype cluster frequencies
	assert L.shape[0] == F.shape[0], "Number of windows doesn't match!"
	assert L.shape[1] == Q.shape[0]*2, "Number of individuals doesn't match!"
	assert Q.shape[1] == F.shape[1], "Number of ancestral components doesn't match!"
	W, N, C = L.shape
	K = Q.shape[1]
	print("Loaded {} haplotypes, {} windows, {} clusters.".format(N, W, C))
	print("Performing FATASH with {} ancestral components.".format(K))

	# Convert log-like to like
	shared_cy.createLikes(L, args.threads)

	# Initiate containers and values
	alpha = args.alpha
	E = np.zeros((W, N, K), dtype=np.float32) # Emission probabilities
	T = np.zeros((K, K), dtype=np.float32) # Transition probabilities
	P = np.zeros((W, N, K), dtype=np.float32) # Posterior
	if args.viterbi:
		V = np.zeros((W, N), dtype=np.int32) # Viterbi path
	if args.alpha_save:
		a = np.ones(N, dtype=np.float32)*alpha # Individual alpha values

	# Compute emissions
	lahmm_cy.calcEmission(L, F, E, args.threads)
	del L, F

	# Run HMM for each individual
	for i in range(N):
		print("\rProcessing haplotype {}/{}".format(i, N), end="")
		# Optimize alpha
		if args.alpha_optim:
			opt=optim.minimize_scalar(fun=loglike_wrapper,
										args=(E, Q[i//2], T, i),
										method='bounded',
										bounds=tuple(args.alpha_bound))
			alpha = opt.x
			if args.alpha_save:
				a[i] = alpha
		# Compute transitions
		lahmm_cy.calcTransition(T, Q[i//2], alpha)

		# Compute posterior probabilities (forward-backward)
		lahmm_cy.fwdbwd(E, Q[i//2], P, T, i)
		if args.viterbi:
			# Viterbi
			lahmm_cy.viterbi(E, Q[i//2], V, T, i)
	print(".")

	# Save matrices
	np.savetxt(args.out + ".path", np.argmax(P, axis=2).T, fmt="%i")
	print("Saved posterior decoding path as " + args.out + ".path")
	np.save(args.out + ".prob", P.astype(float))
	print("Saved posterior probabilities as " + args.out + ".prob")
	if args.viterbi:
		np.savetxt(args.out + ".viterbi", V.T, fmt="%i")
		print("Saved viterbi decoing path as " + args.out + ".viterbi")
	if args.alpha_save:
		np.savetxt(args.out + ".alpha", a, fmt="%.7f")
		print("Saved individual alpha values as " + args.out + ".alpha")
