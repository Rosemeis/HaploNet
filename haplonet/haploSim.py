"""
HaploNet.
Simulating chromosomes using generative GMVAE model.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import own scripts
from haplonet import haploModel

# Main function
def main(args):
	##### HaploNet - Simulation #####
	print("HaploNet - Simulation")
	
	# Check input
	assert (args.clusters is not None), \
		"No input haplotype cluster probabilities (-y)"
	assert os.path.exists(args.models), \
		"No valid path to models provided (-m)"

	# Load data (and concatentate across windows)
	Y = np.load(args.clusters).astype(np.float32)
	print("Loaded haplotype cluster probabilities.")

	# Preparing containers
	nSeg = len(os.listdir(args.models))
	assert nSeg == Y.shape[0], "Number of windows doesn't match model files!"
	X_list = []
	L = torch.empty((nSeg, Y.shape[1]*args.n_sim, Y.shape[2]))
	if args.subsplit > 0: # Subsplit into smaller windows
		Ls = torch.empty((nSeg*args.subsplit, Y.shape[1]*args.n_sim, Y.shape[2]))
	Y_eye = torch.eye(Y.shape[2]) # Cluster labels

	# Simulating chromosomes
	print(f"Simulating {args.n_sim} chromosomes for each of {Y.shape[1]} probability vectors")
	for i in range(nSeg):
		print(f"\rWindow {i+1}/{nSeg}", end="")
		# Load weights and infer model parameters
		weights = torch.load(f"{args.models}/seg{i}.pt")
		h_dim, x_dim = weights["classify.0.weight"].size()
		z_dim, y_dim = weights["prior_m.weight"].size()
		assert Y.shape[2] == y_dim, "Haplotype cluster dimension doesn't match!"

		# Load weights into GMVAE model
		model = haploModel.GMVAENet(x_dim, h_dim, z_dim, y_dim, args.depth, args.temp)
		model.load_state_dict(weights)
		model.eval()

		# Simulate chromsomes using Gumbel-Softmax
		y = torch.from_numpy(Y[i]).tile(args.n_sim,).view(args.n_sim*Y.shape[1], -1)
		g = -torch.log(-torch.log(torch.rand_like(y))) # Gumbel samples
		y_sampled = F.softmax((torch.log(y) + g)/args.temp, dim=1)
		with torch.no_grad():
			x_sim = torch.round(model.simulate(y_sampled))
			l_sim = model.generateLikelihoods(x_sim, Y_eye)
			L[i,:,:] = l_sim.detach()
			if args.subsplit > 0:
				ls_sim = model.subsplitLikelihoods(x_sim, Y_eye, args.subsplit)
				for sub in range(args.subsplit):
					Ls[args.subsplit*i + sub,:,:] = ls_sim[sub].detach()
			X_list.append(x_sim.detach().numpy().T.astype(np.int8, order="C"))
	print("")
		
	# Saving tensors
	np.save(f"{args.out}.loglike", L.numpy())
	print(f"Saved log-likelihoods as {args.out}.loglike.npy")
	if args.subsplit > 0:
		np.save(f"{args.out}.split.loglike", Ls.numpy())
		print(f"Saved subsplit log-likelihoods as {args.out}.split.loglike.npy")
	X = np.concatenate(X_list, axis=0)
	np.save(f"{args.out}.haplotypes", X)
	print(f"Saved simulated chromosomes as {args.out}.haplotypes.npy")
	print("\n")



##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet simulate'!"
