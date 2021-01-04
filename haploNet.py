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
	help="Dimension of hidden layer")
parser.add_argument("-z_dim", type=int, default=64,
	help="Dimension of latent representation")
parser.add_argument("-y_dim", type=int, default=16,
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
parser.add_argument("-debug", action="store_true",
	help="Print losses")
parser.add_argument("-save_models", action="store_true",
	help="Save models")
parser.add_argument("-split", type=float, default=1.0,
	help="Ratio of training/validation")
parser.add_argument("-patience", type=int, default=9,
	help="Patience for validation loss")
parser.add_argument("-threads", type=int,
	help="Number of threads")
parser.add_argument("-workers", type=int, default=0,
	help="Number of workers to use in data-loader")
parser.add_argument("-out",
	help="Output path")
args = parser.parse_args()

### HaploNet ###
print('HaploNet - Gaussian Mixture Variational Autoencoder')

# Global variables
LOG2PI = 2*log(pi)
LOGK = log(args.y_dim)

if args.threads is not None:
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)

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


### Loss functions
# Log-normal pdf for monte carlo samples
def log_normal(z, mu, logvar):
	return -0.5*torch.sum(torch.pow(z - mu, 2)*torch.exp(-logvar) + logvar + \
		LOG2PI, dim=1)

# VAE loss (ELBO)
def elbo(recon_x, x, z, z_m, z_v, p_m, p_v, y, beta):
	rec_loss = torch.sum(F.binary_cross_entropy_with_logits(recon_x, x, reduction='none'), dim=1)
	gau_loss = log_normal(z, z_m, z_v) - log_normal(z, p_m, p_v)
	cat_loss = torch.sum(y*torch.log(torch.clamp(y, min=1e-8)), dim=1) + LOGK
	return torch.mean(rec_loss + gau_loss + beta*cat_loss)


### Training
# Define training step
def train_epoch(train_loader, model, optimizer, beta, device):
	train_loss = 0.0
	for data in train_loader:
		optimizer.zero_grad()
		batch_x = data.to(device)
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
			batch_x = data.to(device)
			recon_x, z, z_m, z_v, p_m, p_v, y = model(batch_x)
			loss = elbo(recon_x, batch_x, z, z_m, z_v, p_m, p_v, y, beta)
			valid_loss += loss.item()
	model.train()
	return valid_loss/len(valid_loader)

# Train models
if (m % args.x_dim) < args.x_dim//2:
	nSeg = m//args.x_dim
else:
	nSeg = ceil(m/args.x_dim)
Z = torch.empty((n, nSeg, args.z_dim)) # Means
V = torch.empty((n, nSeg, args.z_dim)) # Logvars
Y = torch.empty((n, nSeg, args.y_dim)) # Components

for i in range(nSeg):
	st = time()
	print('Chromosome segment {}/{}'.format(i+1, nSeg))
	# Load segment
	if i == (nSeg-1):
		segG = torch.from_numpy(G[(i*args.x_dim):].T.astype(np.float32))
	else:
		segG = torch.from_numpy(G[(i*args.x_dim):((i+1)*args.x_dim)].T.astype(np.float32))

	# Construct sets
	if args.split < 1.0:
		permVec = np.random.permutation(n)
		nTrain = permVec[:int(n*args.split)]
		nValid = permVec[int(n*args.split):]
		trainLoad = DataLoader(segG[nTrain], batch_size=args.bs, shuffle=True, \
								pin_memory=True, num_workers=args.workers)
		validLoad = DataLoader(segG[nValid], batch_size=args.bs)
		patLoss = float('Inf')
		pat = 0
	else:
		# Training set
		trainLoad = DataLoader(segG, batch_size=args.bs, shuffle=True, pin_memory=True, \
								num_workers=args.workers)

	# Define model
	model = haploModel.HaploNet(segG.size(1), args.h_dim, args.z_dim, args.y_dim, args.temp)
	model.to(dev)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Run training and validation
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
				print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, tLoss, vLoss))
			else:
				print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, tLoss))
	print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, tLoss))
	if args.split < 1.0:
		print('Epoch: {}, Valid Loss: {:.4f}'.format(epoch+1, vLoss))
	print('Time elapsed: {:.4f}'.format(time()-st))

	# Save latent and model
	model.to(torch.device('cpu'))
	model.eval()
	with torch.no_grad():
		z, v, y = model.generateLatent(segG)
		Z[:,i,:], V[:,i,:], Y[:,i,:] = z.detach(), v.detach(), y.detach()
	if args.save_models:
		torch.save(model.state_dict(), args.out + '/models/seg' + str(i) + '.pt')

# Saving latent spaces
np.save(args.out + '.z', Z.numpy())
np.save(args.out + '.v', V.numpy())
np.save(args.out + '.y', Y.numpy())
