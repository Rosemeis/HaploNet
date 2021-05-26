# HaploNet
HaploNet is a framework for inferring haplotype and population structure using neural networks in an unsupervised approach for phased haplotypes of whole-genome sequencing (WGS) data. We utilize a variational autoencoder (VAE) framework to learn mappings to and from a low-dimensional latent space in which we will perform indirect clustering of haplotypes with a Gaussian mixture prior (Gaussian Mixture Variational Autoencoder).

### Citation
Preprint: [Haplotype and Population Structure Inference using Neural Networks in Whole-Genome Sequencing Data](https://www.biorxiv.org/content/10.1101/2020.12.28.424587v1.full).

## Install
```bash
git clone https://github.com/Rosemeis/HaploNet.git
cd HaploNet
python setup.py build_ext --inplace
```

### Dependencies
The HaploNet framework relies on the following Python packages that you can install through conda (recommended) or pip:

- [PyTorch](https://pytorch.org/get-started/locally/)
- NumPy
- Cython
- scikit-allel
- SciPy (for PCA)

Follow the link to find more information on how to install PyTorch for your setup (GPU/CPU).

## Usage
For phased haplotype data for a single chromosome as a VCF file of *N* individuals and *M* SNPs, we can simply generate a *2N x M* haplotype matrix using scikit-allel.
```bash
python generateNPYfromVCF.py -vcf chr1.vcf.gz -out chr1
# Saves int8 NumPy matrix in binary format (chr1.npy)
```

HaploNet can now be trained directly on the generated haplotype matrix as follows (using default parameters and on GPU):
```bash
python haploNet.py chr1.npy -cuda -out haplonet
```
HaploNet outputs the neural network log-likelihoods by default which are used to infer global population structure (PCA and admixture). With the '-latent' argument, the parameters of the learnt latent spaces of the GMVAE can be saved as well. See all available options in HaploNet with the following command:
```bash
python haploNet.py -h
```

All the following analyses assume a directory structure with a folder for each chromosome, e.g. *chr1/haplonet.loglike.npy* will be the neural network log-likelihoods for all windows of chromosome 1 inferred by HaploNet.

### Estimate ancestry proportions and haplotype cluster frequencies
The EM algorithm in HaploNet can be run with *K=2* and 64 threads (CPU based).
```bash
python admixNN.py -folder ./ -like haplonet.loglike.npy -K 2 -t 64 -seed 0 -out haplonet.admixture.k2
```

And the admixture proportions can as an example be plotted in R as follows:
```R
library(RcppCNPy)
q <- npyLoad("haplonet.admixture.k2.q.npy")
barplot(t(q), space=0, border=NA, col=c("dodgerblue3", "firebrick2"), xlab="Individuals", ylab="Proportions", main="HaploNet - Admixture")
```

### Infer population structure using PCA
Estimate eigenvectors directly using SVD (recommended for big datasets):
```bash
python pcaNN.py -folder ./ -like haplonet.loglike.npy -t 64 -out haplonet
```
```R
e <- as.matrix(read.table("haplonet.eigenvecs"))
plot(e[,1:2], main="PCA - HaploNet", xlab="PC1", ylab="PC2")
```

Compute the covariance matrix followed by eigendecomposition in R:
```bash
python pcaNN.py -folder ./ -like haplonet.loglike.npy -cov -t 64 -out haplonet
```
```R
C <- as.matrix(read.table("haplonet.cov"))
e <- eigen(C)
plot(e$vectors[,1:2], main="PCA - HaploNet", xlab="PC1", ylab="PC2")
```
