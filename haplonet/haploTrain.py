"""
HaploNet.
Variational Autoencoder framework for modelling haplotype structure.
"""

__author__ = "Jonas Meisner"

# Libraries
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cyvcf2 import VCF
from torch.utils.data import DataLoader
from datetime import datetime
from time import time
from math import ceil, log, pi

# Import own scripts
from haplonet import haploModel
from haplonet import shared_cy

# Global variables
LOG2PI = log(2*pi)

### Loss functions
# Log-normal pdf for monte carlo samples
def log_normal(z, mu, logvar):
	return -0.5*torch.sum(torch.pow(z - mu, 2)*torch.exp(-logvar) + logvar + \
		LOG2PI, dim=1)

# GMVAE loss - negative ELBO
def elbo(recon_x, x, z, z_m, z_v, p_m, p_v, y):
	rec_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
	gau_loss = log_normal(z, z_m, z_v) - log_normal(z, p_m, p_v)
	cat_loss = torch.sum(y*torch.log(torch.clamp(y, min=1e-8)), dim=1) + \
		log(y.size(1))
	return rec_loss, gau_loss.sum(), cat_loss.sum()


### Training functions
# Define GMVAE training step
def train(loader, model, optimizer, beta, device):
	tot_loss = 0.0
	rec_loss = 0.0
	gau_loss = 0.0
	cat_loss = 0.0
	for data in loader:
		optimizer.zero_grad(set_to_none=True)
		batch_x = data.to(device, non_blocking=True)
		recon_x, z, z_m, z_v, p_m, p_v, y = model(batch_x)
		rLoss, gLoss, cLoss = elbo(recon_x, batch_x, z, z_m, z_v, p_m, p_v, y)
		tLoss = rLoss + gLoss + beta*cLoss
		tLoss.backward()
		optimizer.step()
		tot_loss += tLoss.item()
		rec_loss += rLoss.item()
		gau_loss += gLoss.item()
		cat_loss += cLoss.item()
	N = len(loader.dataset)
	return tot_loss/N, rec_loss/N, gau_loss/N, cat_loss/N


### Validation functions
# Define GMVAE validation step
def valid(loader, model, beta, device):
	val_loss = 0.0
	model.eval()
	with torch.no_grad():
		for data in loader:
			batch_x = data.to(device, non_blocking=True)
			recon_x, z, z_m, z_v, p_m, p_v, y = model(batch_x)
			rLoss, gLoss, cLoss = elbo(recon_x, batch_x, z, z_m, z_v, p_m, p_v, y)
			tLoss = rLoss + gLoss + beta*cLoss
			val_loss += tLoss.item()
	model.train()
	return val_loss/len(loader.dataset)

##### Main function #####
def main(args, deaf):
	### HaploNet
	print("HaploNet v0.5")
	print("Gaussian Mixture Variational Autoencoder.")
	assert args.vcf is not None, \
		"Please provide phased genotype file (--bcf or --vcf)!"

	# Create log-file of arguments
	full = vars(args)
	with open(args.out + ".args", "w") as f:
		f.write("HaploNet v0.5\n")
		f.write("haplonet train\n")
		f.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		f.write(f"Directory: {os.getcwd()}\n")
		f.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					f.write(f"\t--{key}\n")
				else:
					f.write(f"\t--{key} {full[key]}\n")
	del full, deaf

	# Setup parameters
	if args.threads is not None:
		torch.set_num_threads(args.threads)
		torch.set_num_interop_threads(args.threads)
	if args.seed is not None:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)

	# Check parsing
	if args.cuda:
		assert torch.cuda.is_available(), "Setup doesn't have GPU support!"
		dev = torch.device("cuda")
	else:
		dev = torch.device("cpu")

	# Create models folder if doesn't exist
	if args.save_models:
		if not os.path.exists(args.out + "/models"):
			os.makedirs(args.out + "/models")


	### Read VCF/BCF file
	print("\rLoading VCF/BCF file...", end="")
	v_file = VCF(args.vcf, threads=args.threads)
	n = 2*len(v_file.samples)
	G = shared_cy.readVCF(v_file, n//2)
	del v_file
	m = G.shape[0]
	print(f"\rLoaded phased genotype data: {n} haplotypes and {m} SNPs.")


	### Construct containers
	if args.windows is None: # Fixed window size
		if (m % args.x_dim) < args.x_dim//2:
			nSeg = m//args.x_dim
		else:
			nSeg = ceil(m/args.x_dim)
		winList = [w*args.x_dim for w in range(nSeg)]
		winList.append(m)
		winList = np.array(winList, dtype=int)
		print(f"Training {nSeg} windows of fixed size ({args.x_dim}).\n")
	else: # Window size based on e.g. LD
		winList = np.genfromtxt(args.windows, dtype=int)
		nSeg = winList.shape[0] - 1
		print(f"Training {nSeg} windows (provided).\n")
	L = torch.empty((nSeg, n, args.y_dim)) # Log-likelihoods
	Y_eye = torch.eye(args.y_dim).to(dev) # Cluster labels
	if args.latent:
		Y = torch.empty((nSeg, n, args.y_dim)) # Components
		Z = torch.empty((nSeg, n, args.z_dim)) # Means
		V = torch.empty((nSeg, n, args.z_dim)) # Logvars
	if args.subsplit > 0: # Subsplit into smaller windows
		Ls = torch.empty((nSeg*args.subsplit, n, args.y_dim))


	### Training
	for i in range(nSeg):
		st = time()
		print(f"Window {i+1}/{nSeg}")

		# Load segment
		segG = torch.from_numpy(G[winList[i]:winList[i+1]].T.astype(np.float32, order="C"))

		# Construct sets
		if args.split < 1.0:
			permVec = np.random.permutation(n)
			nTrain = permVec[:int(n*args.split)]
			nValid = permVec[int(n*args.split):]
			trainLoad = DataLoader(segG[nTrain], batch_size=args.batch, \
				shuffle=True, pin_memory=True)
			validLoad = DataLoader(segG[nValid], batch_size=args.batch, \
				pin_memory=True)
			patLoss = float("Inf")
			pat = 0
		else:
			# Training set
			trainLoad = DataLoader(segG, batch_size=args.batch, shuffle=True, \
				pin_memory=True)

		# Define model
		model = haploModel.GMVAENet(segG.size(1), args.h_dim, args.z_dim, \
			args.y_dim, args.depth, args.temp)
		model.to(dev)
		optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate)

		# Run training (and validation)
		for epoch in range(args.epochs):
			tLoss, rLoss, gLoss, cLoss = train(trainLoad, model, optimizer, args.beta, dev)
			if args.split < 1.0:
				vLoss = valid(validLoad, model, args.beta, dev)
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
					print("Epoch: {}, Train -ELBO: {:.4f}, Valid -ELBO: {:.4f}, Rec: {:.4f}, \
						Gau: {:.4f}, Cat: {:.4f}".format(epoch+1, tLoss, vLoss, rLoss, gLoss, cLoss))
				else:
					print("Epoch: {}, Train -ELBO: {:.4f}, Rec: {:.4f}, Gau: {:.4f}, \
						Cat: {:.4f}".format(epoch+1, tLoss, rLoss, gLoss, cLoss))
		print("Epoch: {}, Train -ELBO: {:.4f}".format(epoch+1, tLoss))
		if args.split < 1.0:
			print("Epoch: {}, Valid -ELBO: {:.4f}".format(epoch+1, vLoss))
		print("Time elapsed: {:.4f}\n".format(time()-st))

		# Generate log-likelihoods and optionally save latent spaces and model
		model.eval()
		batch_n = ceil(n/args.batch)
		saveLoad = DataLoader(segG, batch_size=args.batch, pin_memory=True)
		with torch.no_grad():
			for it, data in enumerate(saveLoad):
				# Generate log-likelihoods
				batch_x = data.to(dev, non_blocking=True)
				batch_l = model.generateLikelihoods(batch_x, Y_eye)
				if it == (batch_n - 1):
					L[i,it*args.batch:,:] = batch_l.to(torch.device("cpu")).detach()
				else:
					L[i,it*args.batch:(it+1)*args.batch,:] = batch_l.to(
						torch.device("cpu")).detach()
				
				# Subsplit log-likelihoods
				if args.subsplit > 0:
					batch_ls = model.subsplitLikelihoods(batch_x, Y_eye, args.subsplit)
					for sub in range(args.subsplit):
						if it == (batch_n - 1):
							Ls[args.subsplit*i + sub, \
								it*args.batch:,:] = batch_ls[sub].to(torch.device("cpu")).detach()
						else:
							Ls[args.subsplit*i + sub, \
								it*args.batch:(it+1)*args.batch,:] = batch_ls[sub].to(\
									torch.device("cpu")).detach()

				# Generate latent spaces
				if args.latent:
					batch_z, batch_v, batch_y = model.generateLatent(batch_x)
					if it == (batch_n - 1):
						Y[i,it*args.batch:,:] = batch_y.to(torch.device("cpu")).detach()
						Z[i,it*args.batch:,:] = batch_z.to(torch.device("cpu")).detach()
						V[i,it*args.batch:,:] = batch_v.to(torch.device("cpu")).detach()
					else:
						Y[i,it*args.batch:(it+1)*args.batch,:] = \
							batch_y.to(torch.device("cpu")).detach()
						Z[i,it*args.batch:(it+1)*args.batch,:] = \
							batch_z.to(torch.device("cpu")).detach()
						V[i,it*args.batch:(it+1)*args.batch,:] = \
							batch_v.to(torch.device("cpu")).detach()
		if args.save_models:
			model.to(torch.device("cpu"))
			torch.save(model.state_dict(), f"{args.out}/models/seg{i}.pt")

		# Release memory
		del segG

	### Saving tensors
	np.save(f"{args.out}.loglike", L.numpy())
	print(f"Saved log-likelihoods as {args.out}.loglike.npy")
	if args.latent:
		np.save(f"{args.out}.y", Y.numpy())
		print(f"Saved Categorical parameters as {args.out}.y.npy")
		np.save(f"{args.out}.z", Z.numpy())
		np.save(f"{args.out}.v", V.numpy())
		print(f"Saved Gaussian parameters as {args.out}.(z,v).npy")
	if args.subsplit > 0:
		np.save(f"{args.out}.split.loglike", Ls.numpy())
		print(f"Saved subsplit log-likelihoods as {args.out}.split.loglike.npy")
	print("\n")



##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet train'!"
