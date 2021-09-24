"""
HaploNet.
Generate .npy file from VCF file.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
import allel
import argparse
from math import ceil

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vcf",
	help="Input vcf-file of genotypes")
parser.add_argument("-l", "--length", metavar="INT", type=int,
	help="Generate median base positions for defined window lengths")
parser.add_argument("-c", "--chromosome", metavar="INT", type=int,
	help="Specify chromosome number to avoid ambiguity")
parser.add_argument("-w", "--windows", action="store_true",
	help="Only save median base positions, no .npy output")
parser.add_argument("-o", "--out",
	help="Output filepath")
parser.add_argument("--unphased", action="store_true",
	help="(Not ready) Toggle for unphased genotype data")
args = parser.parse_args()

# Load VCF
print("Loading VCF file and generating NumPy file.")
vcf = allel.read_vcf(args.vcf)

# Optional save for median window positions (base positions)
if args.length is not None:
	assert args.chromosome is not None, "Please specify chromosome number!"
	print("Generating median window base positions.")
	S = vcf['variants/POS']
	if (S.shape[0] % args.length) < args.length//2:
		nSeg = S.shape[0]//args.length
	else:
		nSeg = ceil(S.shape[0]/args.length)
	M = np.zeros((nSeg, 4), dtype=int)
	for i in range(nSeg):
		if i == (nSeg-1):
			M[i,0] = args.chromosome
			M[i,1] = S[i*args.length]
			M[i,2] = ceil(np.median(S[(i*args.length):]))
			M[i,3] = S[-1]
		else:
			M[i,0] = args.chromosome
			M[i,1] = S[i*args.length]
			M[i,2] = ceil(np.median(S[(i*args.length):((i+1)*args.length)]))
			M[i,3] = S[(i+1)*args.length]
	print("Saving median window base postions as " + args.out + ".median.txt")
	np.savetxt(args.out + ".median.txt", M, delimiter="\t", fmt="%.d")

# Save .npy file in np.int8
if not args.windows:
	G = vcf['calldata/GT']
	if args.unphased:
		print("Saving unphased genotypes as " + args.out + ".npy")
		np.save(args.out, np.sum(G, axis=2).astype(np.int8))
	else:
		print("Saving phased genotypes as " + args.out + ".npy")
		np.save(args.out, G.reshape(G.shape[0], -1))
