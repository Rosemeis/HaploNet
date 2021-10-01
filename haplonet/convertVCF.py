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

# Main function
def main(args):
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
		np.savetxt(args.out + ".median.txt", M, delimiter="\t", fmt="%.d")
		print("Saved median window base postions as " + args.out + ".median.txt")

	# Save .npy file in np.int8
	if not args.windows:
		G = vcf['calldata/GT']
		if args.unphased:
			np.save(args.out, np.sum(G, axis=2).astype(np.int8))
			print("Saved unphased genotypes as " + args.out + ".npy")
		else:
			np.save(args.out, G.reshape(G.shape[0], -1))
			print("Saved phased genotypes as " + args.out + ".npy")


##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet convert'!"
