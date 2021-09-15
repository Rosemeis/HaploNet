"""
HaploNet.
Gaussian Mixture Variational Autoencoder for modelling haplotype structure.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime
from time import time
from math import ceil

# Import own scripts
import haploModel

# Argparse
parser = argparse.ArgumentParser(prog="HaploNet")
parser.add_argument("-g", "--geno",
	help="Genotype file in binary NumPy format")
parser.add_argument("-x", "--x_dim", type=int, default=1024,
	help="Dimension of input data - window size")
parser.add_argument("-i", "--h_dim", type=int, default=256,
	help="Dimension of hidden layers")
parser.add_argument("-z", "--z_dim", type=int, default=64,
	help="Dimension of latent representation")
parser.add_argument("-y", "--y_dim", type=int, default=32,
	help="Number of haplotype clusters")
parser.add_argument("-b", "--batch", type=int, default=128,
	help="Batch size for NN")
parser.add_argument("-e", "--epochs", type=int, default=200,
	help="Number of epochs")
parser.add_argument("-r", "--rate", type=float, default=1e-3,
	help="Learning rate for Adam")
parser.add_argument("-w", "--beta", type=float, default=1.0,
	help="Weight on categorical loss")
parser.add_argument("--temp", type=float, default=0.1,
	help="Temperature in Gumbel-Softmax")
parser.add_argument("-c", "--cuda", action="store_true",
	help="Toggle GPU training")
parser.add_argument("-s", "--seed", type=int,
	help="Set random seed")
parser.add_argument("-l", "--latent", action="store_true",
	help="Save latent space parameters")
parser.add_argument("-p", "--priors", action="store_true",
	help="Save means of priors (E[p(z | y)])")
parser.add_argument("-t", "--threads", type=int,
	help="Number of threads")
parser.add_argument("-o", "--out", default="haplonet",
	help="Output path")
parser.add_argument("--split", type=float, default=1.0,
	help="Ratio of training/validation")
parser.add_argument("--patience", type=int, default=9,
	help="Patience for validation loss")
parser.add_argument("--save_models", action="store_true",
	help="Save models")
parser.add_argument("--debug", action="store_true",
	help="Print losses")
args = parser.parse_args()

##### HaploNet #####
print("HaploNet - Gaussian Mixture Variational Autoencoder")
assert args.geno is not None, "No input data (.npy)"

# Create log-file of arguments
full = vars(parser.parse_args())
deaf = vars(parser.parse_args([]))
with open(args.out + ".args", "w") as f:
	f.write("HaploNet v0.1\n")
	f.write("Time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
	f.write("Directory: " + str(os.getcwd()) + "\n")
	f.write("Options:\n")
	for key in full:
		if full[key] != deaf[key]:
			if type(full[key]) is bool:
				f.write("\t--" + str(key) + "\n")
			else:
				f.write("\t--" + str(key) + " " + str(full[key]) + "\n")
del full, deaf

# Global variables
LOG2PI = 2*np.log(np.pi)
LOGK = np.log(args.y_dim)
if args.threads is not None:
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)
if args.seed is not None:
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

# Check parsing
if args.cuda:
	assert torch.cuda.is_available(), "Setup doesn't have GPU support"
	dev = torch.device("cuda")
else:
	dev = torch.device("cpu")

# Create models folder if doesn't exist
if args.save_models:
	if not os.path.exists(args.out + "/models"):
		os.makedirs(args.out + "/models")

# Read genotype matrix
G = np.load(args.geno)
m, n = G.shape
print("Loaded " + str(n) + " chromosomes and " + str(m) + " variants.")


### Loss functions
# Log-normal pdf for monte carlo samples
def log_normal(z, mu, logvar):
	return -0.5*torch.sum(torch.pow(z - mu, 2)*torch.exp(-logvar) + logvar + \
		LOG2PI, dim=1)

# VAE loss (ELBO)
def elbo(recon_x, x, z, z_m, z_v, p_m, p_v, y, beta):
	rec_loss = torch.sum(F.binary_cross_entropy_with_logits(recon_x, x, reduction="none"), dim=1)
	gau_loss = log_normal(z, z_m, z_v) - log_normal(z, p_m, p_v)
	cat_loss = torch.sum(y*torch.log(torch.clamp(y, min=1e-8)), dim=1) + LOGK
	return torch.mean(rec_loss + gau_loss + beta*cat_loss)


### Training steps
# Define training step
def train_epoch(train_loader, model, optimizer, beta, device):
	train_loss = 0.0
	for data in train_loader:
		optimizer.zero_grad(set_to_none=True)
		batch_x = data.to(device, non_blocking=True)
		recon_x, z, z_m, z_v, p_m, p_v, y = model(batch_x)
		loss = elbo(recon_x, batch_x, z, z_m, z_v, p_m, p_v, y, beta)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
	return train_loss/len(train_loader)

# Define validation step
def valid_epoch(valid_loader, model, beta, device):
	valid_loss = 0.0
	model.eval()
	with torch.no_grad():
		for data in valid_loader:
			batch_x = data.to(device, non_blocking=True)
			recon_x, z, z_m, z_v, p_m, p_v, y = model(batch_x)
			loss = elbo(recon_x, batch_x, z, z_m, z_v, p_m, p_v, y, beta)
			valid_loss += loss.item()
	model.train()
	return valid_loss/len(valid_loader)


### Construct containers
if (m % args.x_dim) < args.x_dim//2:
	nSeg = m//args.x_dim
else:
	nSeg = ceil(m/args.x_dim)
L = torch.empty((nSeg, n, args.y_dim)) # Log-likelihoods
Y_eye = torch.eye(args.y_dim).to(dev) # Cluster labels
if args.latent:
	Z = torch.empty((nSeg, n, args.z_dim)) # Means
	V = torch.empty((nSeg, n, args.z_dim)) # Logvars
	Y = torch.empty((nSeg, n, args.y_dim)) # Components
if args.priors:
	P = torch.empty((nSeg, args.y_dim, args.z_dim)) # Prior means


### Training
print("Training " + str(nSeg) + " windows.\n")
for i in range(nSeg):
	st = time()
	print("Window {}/{}".format(i+1, nSeg))

	# Load segment
	if i == (nSeg-1):
		segG = torch.from_numpy(G[(i*args.x_dim):].T.astype(np.float32, order="C"))
	else:
		segG = torch.from_numpy(G[(i*args.x_dim):((i+1)*args.x_dim)].T.astype(np.float32, order="C"))

	# Construct sets
	if args.split < 1.0:
		permVec = np.random.permutation(n)
		nTrain = permVec[:int(n*args.split)]
		nValid = permVec[int(n*args.split):]
		trainLoad = DataLoader(segG[nTrain], batch_size=args.batch, shuffle=True, \
								pin_memory=True)
		validLoad = DataLoader(segG[nValid], batch_size=args.batch, \
								pin_memory=True)
		patLoss = float("Inf")
		pat = 0
	else:
		# Training set
		trainLoad = DataLoader(segG, batch_size=args.batch, shuffle=True, \
								pin_memory=True)

	# Define model
	model = haploModel.HaploNet(segG.size(1), args.h_dim, args.z_dim, \
								args.y_dim, args.temp)
	model.to(dev)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.rate)

	# Run training (and validation)
	for epoch in range(args.epochs):
		tLoss = train_epoch(trainLoad, model, optimizer, args.beta, dev)
		if args.split < 1.0:
			vLoss = valid_epoch(validLoad, model, args.beta, dev)
			if vLoss > patLoss:
				pat += 1
				if pat == args.patience:
					print("Patience reached.")
					break
			else:
				patLoss = vLoss
				pat = 0
		if args.debug:
			if args.split < 1.0:
				print("Epoch: {}, Train -ELBO: {:.4f}, Valid -ELBO: {:.4f}".format(epoch+1, tLoss, vLoss))
			else:
				print("Epoch: {}, Train -ELBO: {:.4f}".format(epoch+1, tLoss))
	print("Epoch: {}, Train -ELBO: {:.4f}".format(epoch+1, tLoss))
	if args.split < 1.0:
		print("Epoch: {}, Valid -ELBO: {:.4f}".format(epoch+1, vLoss))
	print("Time elapsed: {:.4f}".format(time()-st) + "\n")

	# Save latent and model
	model.eval()
	batch_n = ceil(n/args.batch)
	saveLoad = DataLoader(segG, batch_size=args.batch, pin_memory=True)
	with torch.no_grad():
		for it, data in enumerate(saveLoad):
			# Generate likelihoods
			batch_x = data.to(dev, non_blocking=True)
			batch_l = model.generateLikelihoods(batch_x, Y_eye)
			if it == (batch_n - 1):
				L[i,it*args.batch:,:] = batch_l.to(torch.device("cpu")).detach()
			else:
				L[i,it*args.batch:(it+1)*args.batch,:] = batch_l.to(torch.device("cpu")).detach()

			# Generate latent spaces
			if args.latent:
				batch_z, batch_v, batch_y = model.generateLatent(batch_x)
				if it == (batch_n - 1):
					Z[i,it*args.batch:,:] = batch_z.to(torch.device("cpu")).detach()
					V[i,it*args.batch:,:] = batch_v.to(torch.device("cpu")).detach()
					Y[i,it*args.batch:,:] = batch_y.to(torch.device("cpu")).detach()
				else:
					Z[i,it*args.batch:(it+1)*args.batch,:] = batch_z.to(torch.device("cpu")).detach()
					V[i,it*args.batch:(it+1)*args.batch,:] = batch_v.to(torch.device("cpu")).detach()
					Y[i,it*args.batch:(it+1)*args.batch,:] = batch_y.to(torch.device("cpu")).detach()
		if args.priors:
			p = model.prior_m(Y_eye)
			P[i,:,:] = p.to(torch.device("cpu")).detach()
	if args.save_models:
		model.to(torch.device("cpu"))
		torch.save(model.state_dict(), args.out + "/models/seg" + str(i) + ".pt")

	# Release memory
	del segG

### Saving tensors
np.save(args.out + ".loglike", L.numpy())
print("Saved log-likelihoods as " + args.out + ".loglike.npy")
if args.latent:
	np.save(args.out + ".z", Z.numpy())
	np.save(args.out + ".v", V.numpy())
	np.save(args.out + ".y", Y.numpy())
	print("Saved Gaussian parameters as " + args.out + ".{z,v}.npy")
	print("Saved Categorical parameters as " + args.out + ".y.npy")
if args.priors:
	np.save(args.out + ".z.prior", P.numpy())
	print("Saved prior parameters as " + args.out + ".z.prior.npy")
print("\n")
