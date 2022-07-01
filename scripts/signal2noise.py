import numpy as np
import signal2noise
import argparse

# Argparse
parser = argparse.ArgumentParser(prog="Signal-to-noise")
parser.add_argument("F", help="File for measurements (admixture or PCA")
parser.add_argument("L", help="Numeric labels file (1:K)")
args = parser.parse_args()

F = np.genfromtxt(args.F, dtype=np.float32)
L = np.genfromtxt(args.L, dtype=np.int32)-1
K = max(L)+1
O = np.argsort(L)
F = F[O,:]
L = L[O]

B = 0.0
W = 0.0
for k in range(K):
	b_k = signal2noise.betweenDist(F, L, k)
	w_k = signal2noise.withinDist(F, L, k)
	B += b_k
	W += w_k

print((B/float(K))/(W/float(K)))
