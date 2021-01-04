"""
HaploNet. Generate .npy file from phased VCF file.
"""

import numpy as np
import allel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-vcf")
parser.add_argument("-out")
args = parser.parse_args()

# Load VCF
vcf = allel.read_vcf(args.vcf)
G = vcf['calldata/GT']

# Save .npy file in np.int8
np.save(args.out, G.reshape(G.shape[0], -1))
