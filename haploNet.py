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
from time import time
from math import ceil, log, pi

# Import own scripts
import haploModel

# Argparse
parser = argparse.ArgumentParser(prog="HaploNet")
parser.add_argument("geno",
	help="Genotype file in binary NumPy format")
parser.add_argument("-x_dim", type=int, default=1024,
	help="Dimension of input data - window size")
parser.add_argument("-h_dim", type=int, default=256,
	help="Dimension of hidden layers")
parser.add_argument("-z_dim", type=int, default=64,
	help="Dimension of latent representation")
parser.add_argument("-y_dim", type=int, default=20,
	help="Number of mixing components")
parser.add_argument("-bs", type=int, default=128,
	help="Batch size for NN")
parser.add_argument("-epochs", type=int, default=200,
	help="Number of epochs")
parser.add_argument("-lr", type=float, default=1e-3,
	help="Learning rate for Adam")
parser.add_argument("-beta", type=float, default=0.1,
	help="Weight on categorical loss")
parser.add_argument("-temp", type=float, default=0.1,
	help="Temperature in Gumbel-Softmax")
parser.add_argument("-cuda", action="store_true",
	help="Toggle GPU training")
parser.add_argument("-seed", type=int,
	help="Set NumPy seed")
parser.add_argument("-debug", action="store_true",
	help="Print losses")
parser.add_argument("-save_models", action="store_true",
	help="Save models")
parser.add_argument("-split", type=float, default=1.0,
	help="Ratio of training/validation")
parser.add_argument("-patience", type=int, default=9,
	help="Patience for validation loss")
parser.add_argument("-overlap", action="store_true",
	help="Overlap genomic windows")
parser.add_argument("-latent", action="store_true",
	help="Save latent spaces")
parser.add_argument("-prior", action="store_true",
	help="Save means of priors (E[p(z | y)])")
parser.add_argument("-threads", type=int,
	help="Number of threads")
parser.add_argument("-out",
	help="Output path")
args = parser.parse_args()

##### HaploNet #####
print('HaploNet - Gaussian Mixture Variational Autoencoder')

# Global variables
LOG2PI = 2*log(pi)
if args.threads is not None:
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)
if args.seed is not None:
	np.random.seed(args.seed)

# Check parsing
if args.cuda:
	assert torch.cuda.is_available(), "Setup doesn't have GPU support"
	dev = torch.device("cuda")
else:
	dev = torch.device("cpu")

# Create models folder if doesn't exist
if args.save_models:
	if not os.path.exists(args.out + '/models'):
		os.makedirs(args.out + '/models')

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
	rec_loss = torch.sum(F.binary_cross_entropy_with_logits(recon_x, x, reduction='none'), dim=1)
	gau_loss = log_normal(z, z_m, z_v) - log_normal(z, p_m, p_v)
	cat_loss = torch.sum(y*torch.log(torch.clamp(y, min=1e-8)), dim=1)
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
if args.overlap:
	nSeg = 2*nSeg-1
	print("Training " + str(nSeg) + " overlapping windows.\n")
else:
	print("Training " + str(nSeg) + " windows.\n")
L = torch.empty((nSeg, n, args.y_dim)) # Log-likelihoods
Y_eye = torch.eye(args.y_dim).to(dev) # Cluster labels
if args.latent:
	Z = torch.empty((nSeg, n, args.z_dim)) # Means
	V = torch.empty((nSeg, n, args.z_dim)) # Logvars
	Y = torch.empty((nSeg, n, args.y_dim)) # Components
if args.prior:
	P = torch.empty((nSeg, args.y_dim, args.z_dim)) # Prior means


### Training
for i in range(nSeg):
	st = time()
	print('Window {}/{}'.format(i+1, nSeg))

	# Load segment
	if args.overlap:
		if (i % 2) == 0:
			if i == (nSeg-1):
				segG = torch.from_numpy(G[((i//2)*args.x_dim):].T.astype(np.float32, order="C"))
			else:
				segG = torch.from_numpy(G[((i//2)*args.x_dim):(((i//2)+1)*args.x_dim)].T.astype(np.float32, order="C"))
		else:
			segG = torch.from_numpy(G[((i//2)*args.x_dim + args.x_dim//2):(((i+1)//2)*args.x_dim + args.x_dim//2)].T.astype(np.float32, order="C"))
	else:
		if i == (nSeg-1):
			segG = torch.from_numpy(G[(i*args.x_dim):].T.astype(np.float32, order="C"))
		else:
			segG = torch.from_numpy(G[(i*args.x_dim):((i+1)*args.x_dim)].T.astype(np.float32, order="C"))

	# Construct sets
	if args.split < 1.0:
		permVec = np.random.permutation(n)
		nTrain = permVec[:int(n*args.split)]
		nValid = permVec[int(n*args.split):]
		trainLoad = DataLoader(segG[nTrain], batch_size=args.bs, shuffle=True, \
								pin_memory=True)
		validLoad = DataLoader(segG[nValid], batch_size=args.bs, \
								pin_memory=True)
		patLoss = float('Inf')
		pat = 0
	else:
		# Training set
		trainLoad = DataLoader(segG, batch_size=args.bs, shuffle=True, \
								pin_memory=True)

	# Define model
	model = haploModel.HaploNet(segG.size(1), args.h_dim, args.z_dim, \
								args.y_dim, args.temp)
	model.to(dev)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
				print('Epoch: {}, Train -ELBO: {:.4f}, Valid -ELBO: {:.4f}'.format(epoch+1, tLoss, vLoss))
			else:
				print('Epoch: {}, Train -ELBO: {:.4f}'.format(epoch+1, tLoss))
	print('Epoch: {}, Train -ELBO: {:.4f}'.format(epoch+1, tLoss))
	if args.split < 1.0:
		print('Epoch: {}, Valid -ELBO: {:.4f}'.format(epoch+1, vLoss))
	print('Time elapsed: {:.4f}'.format(time()-st) + '\n')

	# Save latent and model
	model.eval()
	batch_n = ceil(n/args.bs)
	saveLoad = DataLoader(segG, batch_size=args.bs, pin_memory=True)
	with torch.no_grad():
		for it, data in enumerate(saveLoad):
			# Generate likelihoods
			batch_x = data.to(dev, non_blocking=True)
			batch_l = model.generateLikelihoods(batch_x, Y_eye)
			if it == (batch_n - 1):
				L[i,it*args.bs:,:] = batch_l.to(torch.device('cpu')).detach()
			else:
				L[i,it*args.bs:(it+1)*args.bs,:] = batch_l.to(torch.device('cpu')).detach()

			# Generate latent spaces
			if args.latent:
				batch_z, batch_v, batch_y = model.generateLatent(batch_x)
				if it == (batch_n - 1):
					Z[i,it*args.bs:,:] = batch_z.to(torch.device('cpu')).detach()
					V[i,it*args.bs:,:] = batch_v.to(torch.device('cpu')).detach()
					Y[i,it*args.bs:,:] = batch_y.to(torch.device('cpu')).detach()
				else:
					Z[i,it*args.bs:(it+1)*args.bs,:] = batch_z.to(torch.device('cpu')).detach()
					V[i,it*args.bs:(it+1)*args.bs,:] = batch_v.to(torch.device('cpu')).detach()
					Y[i,it*args.bs:(it+1)*args.bs,:] = batch_y.to(torch.device('cpu')).detach()
		if args.prior:
			p = model.prior_m(Y_eye)
			P[i,:,:] = p.to(torch.device('cpu')).detach()
	if args.save_models:
		model.to(torch.device('cpu'))
		torch.save(model.state_dict(), args.out + '/models/seg' + str(i) + '.pt')

	# Release memory
	del segG

### Saving tensors
np.save(args.out + '.loglike', L.numpy())
if args.latent:
	np.save(args.out + '.z', Z.numpy())
	np.save(args.out + '.v', V.numpy())
	np.save(args.out + '.y', Y.numpy())
if args.prior:
	np.save(args.out + '.z.prior', P.numpy())
