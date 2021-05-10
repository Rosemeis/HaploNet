"""
TODO: Not up to date.

HaploNet.
Gaussian Mixture Variational Autoencoder for modelling LD structure.
Script for sampling chromosomes, reconstructing priors or generating likelihoods.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from math import ceil

# Import own scripts
import haploModel

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-models",
	help="Path to models")
parser.add_argument("-x",
	help="Path to training data")
parser.add_argument("-y",
	help="Path to Ys to be sampled")
parser.add_argument("-x_dim", type=int, default=1024,
	help="Window length")
parser.add_argument("-y_dim", type=int, default=16,
	help="Number of haplotype clusters")
parser.add_argument("-n", type=int, default=2,
	help="Number of haplotypes to sample")
parser.add_argument("-like", action="store_true",
	help="Generate log-likelihods")
parser.add_argument("-reconstruct", action="store_true",
	help="Reconstruct input")
parser.add_argument("-prior", action="store_true",
	help="Reconstruct priors")
parser.add_argument("-noise", action="store_true",
	help="Generate noisy samples from prior")
parser.add_argument("-R", action="store_true",
	help="Save new samples in 64-bit float for easy integration in R")
parser.add_argument("-text", action="store_true",
	help="Store NN likelihoods in text-format")
parser.add_argument("-seg", type=int,
	help="Load a single segment number")
parser.add_argument("-out",
	help="Output filepath")
args = parser.parse_args()

# Check parsing
assert args.models is not None, "Must provide path to trained models!"
assert (args.x is not None) or (args.y is not None), \
		"Must provide either data or labels!"
if args.x is not None:
	assert args.like or args.reconstruct or args.prior, "Need to toggle one \
		option at least!"

### HaploNet ###
if args.y is not None:
	print("Generating samples from haplotype label sequence")
	# Haplotype clusters
	Y = torch.from_numpy(np.load(args.y).astype(int)).unsqueeze(1)
	nSeg = Y.size(0)
	Y_one = torch.zeros((nSeg, args.y_dim))
	Y_one.scatter_(1, Y, 1)

if args.x is not None:
	assert args.x_dim is not None, "Need to specify window size used!"
	# Load data
	X = np.load(args.x)
	m, n = X.shape
	Y_eye = torch.eye(args.y_dim)
	if (m % args.x_dim) < args.x_dim//2:
		nSeg = m//args.x_dim
	else:
		nSeg = ceil(m/args.x_dim)

	# Generate log-likelihoods
	if args.like:
		print("Generating NN log-likelihoods")
		if args.seg is None:
			L = torch.empty((nSeg, n, args.y_dim))

	# Reconstruct input matrix
	if args.reconstruct:
		print("Reconstructing input data matrix")
		if args.seg is None:
			X_R = np.empty((n, m), dtype(np.int8))

	# Prior reconstruction
	if args.prior:
		print("Reconstructing priors")
		if args.seg is None:
			X_P = torch.empty((args.y_dim, m))

# Run through segments
if args.seg is not None:
	W = torch.load(args.models + "/seg" + str(args.seg) + ".pt")
	h_dim, x_dim = W['classify.0.weight'].size()
	z_dim =	W['encoder_m.weight'].size(0)
	y_dim = W['classify.3.weight'].size(0)
	assert y_dim == args.y_dim, "Number of clusters must match with trained model!"
	model = haploModel.HaploNet(x_dim, h_dim, z_dim, y_dim, 0.1)
	model.load_state_dict(W)
	model.eval()
	with torch.no_grad():
		# Generate new samples from priors
		if args.y is not None:
			X_rec = model.samplePrior(Y_one.unsqueeze(0).repeat(args.n, 1), \
										args.noise).detach().numpy()
		# Reconstruct priors or generate likelihoods
		if args.x is not None:
			x_window = torch.from_numpy(X[(args.seg*args.x_dim):((args.seg+1)*args.x_dim)].T.astype(np.float32))
			if args.reconstruct:
				tmp = model.reconstructX(x_window).detach().numpy()
				tmp[tmp < 0.5] = 0.0
				tmp[tmp >= 0.5] = 1.0
				X_R = tmp.astype(np.int8)
			if args.prior:
				X_P = model.samplePrior(Y_eye).detach()
			if args.like:
				L = model.generateLikelihoods(x_window, Y_eye).detach()
else:
	for i in range(nSeg):
		W = torch.load(args.models + "/seg" + str(i) + ".pt")
		h_dim, x_dim = W['classify.0.weight'].size()
		z_dim =	W['encoder_m.weight'].size(0)
		y_dim = W['classify.3.weight'].size(0)
		assert y_dim == args.y_dim, "Number of clusters must match with trained model!"
		model = haploModel.HaploNet(x_dim, h_dim, z_dim, y_dim, 0.1)
		model.load_state_dict(W)
		model.eval()
		with torch.no_grad():
			# Generate new samples from priors
			if args.y is not None:
				x_rec = model.samplePrior(Y_one[i].unsqueeze(0).repeat(args.n, 1), \
											args.noise).detach().numpy()
				if i > 0:
					X_rec = np.concatenate((X_rec, x_rec), axis=1)
				else:
					X_rec = x_rec.copy()
			# Reconstruct priors or generate likelihoods
			if args.x is not None:
				if i == (nSeg - 1):
					x_window = torch.from_numpy(X[(i*args.x_dim):].T.astype(np.float32))
					if args.reconstruct:
						tmp = model.reconstructX(x_window).detach().numpy()
						tmp[tmp < 0.5] = 0.0
						tmp[tmp >= 0.5] = 1.0
						X_R[:,(i*args.x_dim):((i+1)*args.x_dim)] = tmp.astype(np.int8)
					if args.prior:
						X_P[:,(i*args.x_dim):] = model.samplePrior(Y_eye).detach()
				else:
					x_window = torch.from_numpy(X[(i*args.x_dim):((i+1)*args.x_dim)].T.astype(np.float32))
					if args.reconstruct:
						tmp = model.reconstructX(x_window).detach().numpy()
						tmp[tmp < 0.5] = 0.0
						tmp[tmp >= 0.5] = 1.0
						X_R[:,(i*args.x_dim):((i+1)*args.x_dim)] = tmp.astype(np.int8)
					if args.prior:
						X_P[:,(i*args.x_dim):((i+1)*args.x_dim)] = model.samplePrior(Y_eye).detach()
				if args.like:
					L[i,:,:] = model.generateLikelihoods(x_window, Y_eye).detach()

# Save matrices
if args.y is not None:
	if args.R:
		np.save(args.out, X_rec.astype(float))
	else:
		np.save(args.out, X_rec)

if args.x is not None:
	if args.like:
		if args.text and (args.seg is None):
			np.savetxt(args.out + '.loglike', L.numpy().reshape(L.shape[0], -1), fmt='%.6f')
		else:
			np.save(args.out  + '.loglike', L.numpy())
	if args.reconstruct:
		if args.R:
			np.save(args.out + '.recon.x', X_R.astype(float))
		else:
			np.save(args.out + '.recon.x', X_R)
	if args.prior:
		if args.R:
			np.save(args.out + '.recon.prior', X_P.numpy().astype(float))
		else:
			np.save(args.out + '.recon.prior', X_P.numpy())
