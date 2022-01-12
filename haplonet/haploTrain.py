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
from haplonet import haploModel

# Main function
def main(args, deaf):
	##### HaploNet #####
	print("HaploNet v0.2")
	print("Gaussian Mixture Variational Autoencoder.")
	assert (args.geno is not None) or (args.vcf is not None), \
			"No input data (--geno or --vcf)!"

	# Create log-file of arguments
	full = vars(args)
	with open(args.out + ".args", "w") as f:
		f.write("HaploNet v0.2\n")
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
	LOG2PI = np.log(2*np.pi)
	LOGK = np.log(args.y_dim)
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


	### Read genotype matrix or VCF-file
	print("\rLoading data...", end="")
	if args.geno is not None:
		G = np.load(args.geno)
	else:
		import allel
		vcf = allel.read_vcf(args.vcf)
		G = vcf['calldata/GT'].reshape(vcf['calldata/GT'].shape[0], -1)
	m, n = G.shape
	print("\rLoaded {} chromosomes and {} variants.".format(n, m))


	### Loss functions
	# Log-normal pdf for monte carlo samples
	def log_normal(z, mu, logvar):
		return -0.5*torch.sum(torch.pow(z - mu, 2)*torch.exp(-logvar) + logvar + \
			LOG2PI, dim=1)

	# GMVAE loss - negative ELBO
	def elbo(recon_x, x, z, z_m, z_v, p_m, p_v, y):
		rec_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
		gau_loss = log_normal(z, z_m, z_v) - log_normal(z, p_m, p_v)
		cat_loss = torch.sum(y*torch.log(torch.clamp(y, min=1e-8)), dim=1) + LOGK
		return rec_loss, gau_loss.sum(), cat_loss.sum()


	### Training steps
	# Define training step
	def train_epoch(train_loader, model, optimizer, beta, device):
		tot_loss = 0.0
		rec_loss = 0.0
		gau_loss = 0.0
		cat_loss = 0.0
		for data in train_loader:
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
		N = len(train_loader.dataset)
		return tot_loss/N, rec_loss/N, gau_loss/N, cat_loss/N

	# Define validation step
	def valid_epoch(valid_loader, model, beta, device):
		val_loss = 0.0
		model.eval()
		with torch.no_grad():
			for data in valid_loader:
				batch_x = data.to(device, non_blocking=True)
				recon_x, z, z_m, z_v, p_m, p_v, y = model(batch_x)
				rLoss, gLoss, cLoss = elbo(recon_x, batch_x, z, z_m, z_v, p_m, p_v, y)
				tLoss = rLoss + gLoss + beta*cLoss
				val_loss += tLoss.item()
		model.train()
		return val_loss/len(valid_loader.dataset)


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


	### Training
	print("Training {} windows.\n".format(nSeg))
	for i in range(nSeg):
		st = time()
		print("Window {}/{}".format(i+1, nSeg))

		# Load segment
		if i == 0:
			segG = torch.from_numpy(G[(i*args.x_dim):((i+1)*args.x_dim + args.overlap)].T.astype(np.float32, order="C"))
		elif i == (nSeg-1):
			segG = torch.from_numpy(G[(i*args.x_dim - args.overlap):].T.astype(np.float32, order="C"))
		else:
			segG = torch.from_numpy(G[(i*args.x_dim - args.overlap):((i+1)*args.x_dim + args.overlap)].T.astype(np.float32, order="C"))

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
		optimizer = torch.optim.Adam(model.parameters(), lr=args.rate)

		# Run training (and validation)
		for epoch in range(args.epochs):
			tLoss, rLoss, gLoss, cLoss = train_epoch(trainLoad, model, optimizer, args.beta, dev)
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
					print("Epoch: {}, Train -ELBO: {:.4f}, Valid -ELBO: {:.4f}, Rec: {:.4f}, Gau: {:.4f}, Cat: {:.4f}".format(epoch+1, tLoss, vLoss, rLoss, gLoss, cLoss))
				else:
					print("Epoch: {}, Train -ELBO: {:.4f}, Rec: {:.4f}, Gau: {:.4f}, Cat: {:.4f}".format(epoch+1, tLoss, rLoss, gLoss, cLoss))
		print("Epoch: {}, Train -ELBO: {:.4f}".format(epoch+1, tLoss))
		if args.split < 1.0:
			print("Epoch: {}, Valid -ELBO: {:.4f}".format(epoch+1, vLoss))
		print("Time elapsed: {:.4f}\n".format(time()-st))

		# Generate likelihoods and optionally save latent spaces and model
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
		print("Saved Gaussian parameters as " + args.out + ".{z,v}.npy")
		np.save(args.out + ".y", Y.numpy())
		print("Saved Categorical parameters as " + args.out + ".y.npy")
	print("\n")


##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet train'!"
