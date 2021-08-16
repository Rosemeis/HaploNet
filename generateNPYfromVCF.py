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
    help="(Not ready) Toggle for unphased genotype data")
parser.add_argument("-window_length", metavar="INT", type=int,
    help="Generate median base positions for defined window lengths")
parser.add_argument("-only_lengths", action="store_true",
    help="Only save median base positions")
parser.add_argument("-out",
    help="Output filepath")
args = parser.parse_args()

# Load VCF
print("Loading VCF file and generating NumPy file.")
vcf = allel.read_vcf(args.vcf)

# Optional save for median window positions (base positions)
if args.window_length is not None:
    print("Generating median window base positions.")
    S = vcf['variants/POS']
    if (S.shape[0] % args.window_length) < args.window_length//2:
        nSeg = S.shape[0]//args.window_length
    else:
        nSeg = np.ceil(S.shape[0]/args.window_length)
    M = np.zeros((nSeg, 3), dtype=int)
    for i in range(nSeg):
        if i == (nSeg-1):
            M[i,0] = S[i*args.window_length]
            M[i,1] = np.ceil(np.median(S[(i*args.window_length):]))
            M[i,2] = S.shape[0]
        else:
            M[i,0] = S[i*args.window_length]
            M[i,1] = np.ceil(np.median(S[(i*args.window_length):((i+1)*args.window_length)]))
            M[i,2] = S[(i+1)*args.window_length]
    print("Saving median window base postions as " + args.out + ".median.txt")
    np.savetxt(args.out + ".median.txt", M, delimiter="\t")

# Save .npy file in np.int8
if not args.only_lengths:
    G = vcf['calldata/GT']
    if args.unphased:
        print("Saving unphased genotypes as " + args.out + ".npy")
        np.save(args.out, np.sum(G, axis=2).astype(np.int8))
    else:
        print("Saving phased genotypes as " + args.out + ".npy")
        np.save(args.out, G.reshape(G.shape[0], -1))
