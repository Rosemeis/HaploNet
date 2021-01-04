"""
HaploNet.
Model definition of Gaussian Mixture Variational Autoencoder using PyTorch.
"""

__author__ = "Jonas Meisner"

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

##### HaploNet ######
class HaploNet(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, y_dim, temp):
		super(HaploNet, self).__init__()
		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.y_dim = y_dim
		self.temp = temp

		# Classification - q(y | x)
		self.classify = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ELU(),
			nn.BatchNorm1d(h_dim),
			nn.Linear(h_dim, y_dim)
		)

		# Encoder - q(z | x, y)
		self.encoder = nn.Sequential(
			nn.Linear(x_dim + y_dim, h_dim),
			nn.ELU(),
			nn.BatchNorm1d(h_dim)
		)
		self.encoder_m = nn.Linear(h_dim, z_dim)
		self.encoder_v = nn.Linear(h_dim, z_dim)

		# Prior - p(z | y)
		self.prior_m = nn.Linear(y_dim, z_dim)
		self.prior_v = nn.Linear(y_dim, z_dim)

		# Decoder - p(x | z)
		self.decoder = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ELU(),
			nn.BatchNorm1d(h_dim),
			nn.Linear(h_dim, x_dim)
		)

	# Gaussian reparameterization
	def reparameterize_gaussian(self, mu, logvar):
		eps = torch.randn_like(logvar)
		return mu + torch.exp(0.5*logvar)*eps

	# Gumbel-softmax reparameterization
	def reparameterize_gumbel(self, logits, temp):
		return F.gumbel_softmax(logits, tau=temp)

	# Forward propagation
	def forward(self, x):
		y_logits = self.classify(x)
		y = self.reparameterize_gumbel(y_logits, self.temp)
		e = self.encoder(torch.cat((x, y), dim=1))
		z_m = self.encoder_m(e)
		z_v = self.encoder_v(e)
		z = self.reparameterize_gaussian(z_m, z_v)
		p_m = self.prior_m(y)
		p_v = self.prior_v(y)
		return self.decoder(z), z, z_m, z_v, \
				p_m, p_v, F.softmax(y_logits, dim=1)

	# Generate mu and y
	def generateLatent(self, x):
		y = F.softmax(self.classify(x), dim=1)
		e = self.encoder(torch.cat((x, y), dim=1))
		return self.encoder_m(e), self.encoder_v(e), y

	# Generate likelihoods - p(x | y) \propto p(x | p_m(y))
	def generateLikelihoods(self, x, eye):
		return torch.stack([-torch.sum(F.binary_cross_entropy_with_logits( \
							self.decoder(self.prior_m(eye[i].repeat(x.size(0), 1))), x, reduction='none'), dim=1) \
							for i in range(self.y_dim)], dim=1)

	# Sample and reconstruct
	def samplePrior(self, y, noise=False):
		p_m = self.prior_m(y)
		if noise:
			p_v = self.prior_v(y)
			z = self.reparameterize_gaussian(p_m, p_v)
			return torch.sigmoid(self.decoder(z))
		else:
			return torch.sigmoid(self.decoder(p_m))

	# Reconstruct input
	def reconstructX(self, x):
		y_logits = self.classify(x)
		y = F.softmax(y_logits, dim=1)
		e = self.encoder(torch.cat((x, y), dim=1))
		z = self.encoder_m(e)
		return torch.sigmoid(self.decoder(z))
