# HaploNet
HaploNet is a framework for inferring haplotype and population structure using neural networks in an unsupervised approach for phased haplotypes of whole-genome sequencing (WGS) data. We utilize a variational autoencoder (VAE) framework to learn mappings to and from a low-dimensional latent space in which we will perform indirect clustering of haplotypes with a Gaussian mixture prior (Gaussian Mixture Variational Autoencoder).

### Citation
Preprint: [Haplotype and Population Structure Inference using Neural Networks in Whole-Genome Sequencing Data](https://www.biorxiv.org/content/10.1101/2020.12.28.424587v2).

### Dependencies
The HaploNet framework relies on the following Python packages that you can install through conda (recommended) or pip:

- [pytorch](https://pytorch.org/get-started/locally/)
- numpy
- cython
- scipy
- scikit-allel

Follow the link to find more information on how to install PyTorch for your setup (GPU/CPU). You can create an environment through conda easily as follows:
```
conda env create -f environment.yml
```

## Install and build
```bash
git clone https://github.com/Rosemeis/HaploNet.git
cd HaploNet
python setup.py build_ext --inplace
pip3 install -e .
```

You can now run HaploNet with the `haplonet` command.

## Usage
For phased haplotype data for a single chromosome as a VCF file of *M* SNPs and *N* individuals, we can generate a *M x 2N* haplotype matrix using scikit-allel.
```bash
haplonet convert --vcf chr1.vcf.gz --out chr1
# Saves int8 NumPy matrix in binary format (chr1.npy)
```

HaploNet can now be trained directly on the generated haplotype matrix as follows (using default parameters and on GPU):
```bash
haplonet train --geno chr1.npy --cuda --out haplonet

# You can also use the VCF directly as input
haplonet train --vcf chr1.vcf.gz --cuda --out haplonet
```
HaploNet outputs the neural network log-likelihoods by default which are used to infer global population structure (PCA and admixture). With the '--latent' argument, the parameters of the learnt latent spaces of the GMVAE can be saved as well. See all available options in HaploNet with the following command:
```bash
haplonet -h
haplonet train -h # training haplonet
haplonet admix -h # estimate ancestry
haplonet pca -h # perform pca
haplonet convert -h # convert VCF file to NumPy binary format
```

All the following analyses assume that HaploNet has been run for all chromosomes such that the filepaths of the log-likelihoods for each chromosomes are in one file. The argument "--like" can be used if you only have one chromosome or merged file.

### Estimate ancestry proportions and haplotype cluster frequencies
The EM algorithm in HaploNet can be run with *K=2* and 64 threads (CPU based).
```bash
haplonet admix --filelist chr.loglike.list --K 2 --threads 64 --seed 0 --out haplonet.admixture.k2
```

And the admixture proportions can as an example be plotted in R as follows:
```R
q <- read.table("haplonet.admixture.k2.q")
barplot(t(q), space=0, border=NA, col=c("dodgerblue3", "firebrick2"), xlab="Individuals", ylab="Proportions", main="HaploNet - Admixture")
```

### Infer population structure using PCA
Estimate eigenvectors directly using SVD (recommended for big datasets):
```bash
haplonet pca --filelist chr.loglike.list --threads 64 --out haplonet
```
```R
e <- as.matrix(read.table("haplonet.eigenvecs"))
plot(e[,1:2], main="HaploNet - PCA", xlab="PC1", ylab="PC2")
```

Compute the covariance matrix followed by eigendecomposition in R:
```bash
haplonet pca --filelist chr.loglike.list --cov --threads 64 --out haplonet
```
```R
C <- as.matrix(read.table("haplonet.cov"))
e <- eigen(C)
plot(e$vectors[,1:2], main="HaploNet - PCA", xlab="PC1", ylab="PC2")
```
