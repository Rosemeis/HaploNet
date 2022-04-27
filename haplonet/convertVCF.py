"""
HaploNet.
Generate .npy file from VCF file and generate window statistics.
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

	# Optional save for window positions (base positions)
	if args.length is not None:
		print("Generating start and end base positions for windows.")
		S = vcf['variants/POS']
		if (S.shape[0] % args.length) < args.length//2:
			nSeg = S.shape[0]//args.length
		else:
			nSeg = ceil(S.shape[0]/args.length)
		M = np.empty((nSeg, 3), dtype=object)
		M[:,0] = vcf['variants/CHROM'] 
		for i in range(nSeg):
			if i == (nSeg-1):
				M[i,1] = S[i*args.length]
				M[i,2] = S[-1]
			else:
				M[i,1] = S[i*args.length]
				M[i,2] = S[(i+1)*args.length]
		np.savetxt(args.out + ".positions.txt", M, delimiter="\t", fmt="%s")
		print("Saved window base postions as " + args.out + ".positions.txt")

	# Save .npy file in np.int8
	if not args.windows:
		G = vcf['calldata/GT']
		np.save(args.out, G.reshape(G.shape[0], -1))
		print("Saved phased genotypes as " + args.out + ".npy")


##### Main exception #####
assert __name__ != "__main__", "Please use 'haplonet convert'!"
