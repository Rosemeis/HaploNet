"""
HaploNet.
HMM for inferring local ancestry tracts.
"""

__author__ = "Jonas Meisner and Kristian Hanghoej"

# Libraries
import numpy as np
import scipy.optimize as optim
from haplonet import lahmm_cy
from haplonet import shared_cy

# Log-likehood wrapper for SciPy
def loglike_wrapper(param, *args):
	E, Qi, T, Wi, i = args
	lahmm_cy.calcTransitionDist(T, Qi, Wi, param)
	return lahmm_cy.loglike(E, Qi, T, i)


# Main function
def main(args):
	##### HaploNet - FATASH #####
	print("HaploNet - FATASH")
	print("HMM for inferring local ancestry tracts.\n")

	# Check input
	assert (args.filelist is not None) or (args.like is not None), \
			"No log-likelihoods provided (-f or -l)"
	assert args.prop is not None, "No ancestry proportions provided (-q)"
	assert args.freq is not None, "No haplotype cluster frequencies provided (-p)"

	# Load data
	print("Loading data.")
	if args.filelist is not None:
		L_list = []
		F_list = [0]
		with open(args.filelist) as f:
			file_c = 1
			for chr in f:
				L_list.append(np.load(chr.strip("\n")))
				F_list.append(L_list[-1].shape[0] + F_list[-1])
				print(f"\rParsed file #{file_c}", end="")
				file_c += 1
			print(".")
		L = np.concatenate(L_list, axis=0)
		del L_list
	else:
		L = np.load(args.like)
		
	Q = np.genfromtxt(args.prop).astype(np.float32) # Ancestry proportions
	np.clip(Q, a_min=args.q_bound, a_max=1-args.q_bound, out=Q)
	Q /= Q.sum(1, keepdims=True)
	F = np.load(args.freq).astype(np.float32) # Ancestral haplotype cluster frequencies

	W, N, C = L.shape
	K = Q.shape[1]
	if not args.windows:
		print("Discrete HMM")
		windows = np.ones((N, W), dtype=np.float32) # setting all windows to 1 disables continuous
	else:
		print("Continuous HMM")
		windows = np.loadtxt(args.windows).astype(np.float32)
		assert windows.shape[0] == W, "Number of position windows do not match loglike windows"
		if windows.ndim==2:
			print("Window distances per haplotype")
			assert windows.shape[1] == N, \
				"Number of columns in windows must match number of haplotypes"
		else:
			print("Same window distances for all haplotypes")
			windows = np.repeat(windows, N).reshape(W,N) 
		windows = np.ascontiguousarray(windows.T)
		
	assert L.shape[0] == F.shape[0], "Number of windows doesn't match!"
	assert L.shape[1] == Q.shape[0]*2, "Number of individuals doesn't match!"
	assert Q.shape[1] == F.shape[1], "Number of ancestral components doesn't match!"
	print(f"Loaded {N} haplotypes, {W} windows, {C} clusters.")
	print(f"Performing FATASH with {K} ancestral components.")

	# Convert log-like to like
	shared_cy.createLikes(L, args.threads)

	# Initiate containers and values
	alpha = args.alpha
	E = np.zeros((W, N, K), dtype=np.float32) # Emission probabilities
	T = np.zeros((W, K, K), dtype=np.float32) # Transition probabilities with distances
	if args.alpha_save:
		a = np.ones(N, dtype=np.float32)*alpha # Individual alpha values

	# Compute emissions
	lahmm_cy.calcEmission(L, F, E, args.threads)
	del L, F

	# Single or multiple chromosomes
	if args.filelist is not None:
		n_chr = len(F_list) - 1
	else:
		n_chr = 1

	if args.window_save:
		np.savetxt(f"{args.out}.windows", windows, fmt="%.7f")
		print(f"Saved normalized windows to {args.out}.windows")

	# Run through each chromosome
	for chr in range(n_chr):
		print(f"Chromosome {chr+1}/{n_chr}")
		if n_chr == 1:
			F_list = [0,W]
		P = np.zeros((F_list[chr+1] - F_list[chr], N, K), dtype=np.float32) # Posterior
		if args.viterbi:
			V = np.zeros((F_list[chr+1] - F_list[chr], N), dtype=np.int32) # Viterbi path

		# Run HMM for each individual
		for i in range(N):
			print(f"\rProcessing haplotype {i+1}/{N}", end="")

			# Optimize alpha
			if args.optim:
				opt=optim.minimize_scalar(
					fun=loglike_wrapper,
					args=(E[F_list[chr]:F_list[chr+1]], Q[i//2], T, windows[i], i),
					method='bounded',
					bounds=tuple(args.alpha_bound)
				)
				alpha = opt.x
				if args.alpha_save:
					a[i] = alpha

			# Compute transitions
			lahmm_cy.calcTransitionDist(T, Q[i//2], windows[i], alpha)

			# Compute posterior probabilities (forward-backward)
			lahmm_cy.fwdbwd(E[F_list[chr]:F_list[chr+1]], Q[i//2], P, T, i)
			if args.viterbi:
				# Viterbi
				lahmm_cy.viterbi(E[F_list[chr]:F_list[chr+1]], Q[i//2], V, T, i)
		print(".")

		# Save matrices
		if n_chr == 1:
			np.savetxt(f"{args.out}.path", np.argmax(P, axis=2).T, fmt="%i")
			print(f"Saved posterior decoding path as {args.out}.path")
			np.save(f"{args.out}.prob", P.astype(float))
			print(f"Saved posterior probabilities as {args.out}.prob.npy")
			if args.viterbi:
				np.savetxt(f"{args.out}.viterbi", V.T, fmt="%i")
				print(f"Saved viterbi decoing path as {args.out}.viterbi")
			if args.alpha_save:
				np.savetxt(f"{args.out}.alpha", a, fmt="%.7f")
				print(f"Saved individual alpha values as {args.out}.alpha")
		else:
			np.savetxt(f"{args.out}.chr{chr+1}.path", np.argmax(P, axis=2).T, fmt="%i")
			print(f"Saved posterior decoding path as {args.out}.chr{chr+1}.path")
			np.save(f"{args.out}.chr{chr+1}.prob", P.astype(float))
			print(f"Saved posterior probabilities as {args.out}.chr{chr+1}.prob")
			if args.viterbi:
				np.savetxt(args.out + ".chr{chr+1}.viterbi", V.T, fmt="%i")
				print(f"Saved viterbi decoing path as {args.out}.chr{chr+1}.viterbi")
			if args.alpha_save:
				np.savetxt(args.out + ".chr{chr+1}.alpha", a, fmt="%.7f")
				print(f"Saved individual alpha values as {args.out}.chr{chr+1}.alpha")
		del P
		if args.viterbi:
			del V
