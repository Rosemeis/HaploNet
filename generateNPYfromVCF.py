"""
HaploNet.
Generate .npy file from VCF file.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import allel
import argparse

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-vcf",
    help="Input vcf-file of genotypes")
parser.add_argument("-unphased", action="store_true",
    help="Toggle for unphased genotype data")
parser.add_argument("-out",
    help="Output filepath")
args = parser.parse_args()

# Load VCF
print("Loading VCF file and generating NumPy file.")
vcf = allel.read_vcf(args.vcf)
G = vcf['calldata/GT']

# Save .npy file in np.int8
if args.unphased:
    print("Saving unphased genotypes as " + args.out + ".npy")
    np.save(args.out, np.sum(G, axis=2).astype(np.int8))
else:
    print("Saving phased genotypes as " + args.out + ".npy")
    np.save(args.out, G.reshape(G.shape[0], -1))
