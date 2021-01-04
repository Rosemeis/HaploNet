# HaploNet
Framework. Gaussian Mixture Variational Autoencoder (GMVAE).

*Support for unphased genotypes is coming soon.*

### Citation
Preprint: [Haplotype and Population Structure Inference using Neural Networks in Whole-Genome Sequencing Data](https://www.biorxiv.org/content/10.1101/2020.12.28.424587v1.full).

## Install
```bash
git clone https://github.com/Rosemeis/HaploNet.git
cd HaploNet
python setup.py build_ext --inplace
```

### Dependencies
The HaploNet framework relies on the following Python packages that you can install through pip or conda:

- [PyTorch](https://pytorch.org/get-started/locally/)
- NumPy
- scikit-allel

Follow the link to find more information on how to install PyTorch for your setup (GPU/CPU).

## Usage
For phased haplotype data for a single chromosome as a VCF file of *N* individuals and *M* SNPs, we can simply generate a *2N x M* haplotype matrix using scikit-allel.
```bash
python generateNPYfromVCF.py -vcf chr1.vcf.gz -out chr1
# Saves int8 NumPy matrix in binary format (chr1.npy)
```

HaploNet can now be trained directly on the generated haplotype matrix as follows (using default parameters and on GPU):
```bash
python haploNet.py chr1.npy -cuda -save_models -out haplonet
```
The '-save_models' parameter is needed for some downstream analysis. See available options in HaploNet with the following command:
```bash
python haploNet.py -h
```

All the following analyses assume a directory structure with a folder for each chromosome, e.g. *chr1/haplonet.z.npy* will be the mean latent spaces for windows of chromosome 1 inferred by HaploNet.

### Infer population structure using PCA
Compute the covariance matrix followed by eigendecomposition in R (using the RcppCNPy library):
```bash
python computeDist.py -folder ./ -latent haplonet -out haplonet
```
```R
library(RcppCNPy)
C <- npyLoad("haplonet.cov.npy")
e <- eigen(C)
plot(e$vectors[,1:2], main="PCA - HaploNet", xlab="PC1", ylab="PC2")
```

### Estimate ancestry proportions
First the neural network log-likelihoods have to be generated for each chromosome. Then the admixture proportions can be estimated afterwards, e.g. *K=2* using 64 threads.
```bash
python sampling.py -models chr1/haplonet/models/ -x chr1/chr1.npy -like -out haplonet
python admixNN.py -folder ./ -like haplonet.loglike.npy -K 2 -t 64 -seed 1 -out haplonet.admixture
```
