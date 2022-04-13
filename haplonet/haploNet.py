"""
Main caller of the HaploNet framework.
"""

import argparse
import sys

# Main function
def main():
	# Argparser
	parser = argparse.ArgumentParser(prog="haplonet")
	subparsers = parser.add_subparsers(title="HaploNet commands")

	### Commands
	# haplonet train
	parser_t = subparsers.add_parser("train")
	parser_t.add_argument("-g", "--geno",
		help="Genotype file in binary NumPy format")
	parser_t.add_argument("-v", "--vcf",
		help="Genotype file in VCF format")
	parser_t.add_argument("-x", "--x_dim", type=int, default=1024,
		help="Dimension of input data - window size (1024)")
	parser_t.add_argument("-i", "--h_dim", type=int, default=256,
		help="Dimension of hidden layers (256)")
	parser_t.add_argument("-z", "--z_dim", type=int, default=64,
		help="Dimension of latent representation (64)")
	parser_t.add_argument("-y", "--y_dim", type=int, default=32,
		help="Number of haplotype clusters (32)")
	parser_t.add_argument("-b", "--batch", type=int, default=128,
		help="Batch size for NN (128)")
	parser_t.add_argument("-e", "--epochs", type=int, default=200,
		help="Number of epochs (200)")
	parser_t.add_argument("-r", "--rate", type=float, default=1e-3,
		help="Learning rate for Adam (1e-3)")
	parser_t.add_argument("-s", "--seed", type=int,
		help="Set random seed")
	parser_t.add_argument("-c", "--cuda", action="store_true",
		help="Toggle GPU training")
	parser_t.add_argument("-t", "--threads", type=int,
		help="Number of threads")
	parser_t.add_argument("-o", "--out", default="haplonet",
		help="Output path/name ('haplonet')")
	parser_t.add_argument("--beta", type=float, default=1.0,
		help="Weight on categorical loss (1.0)")
	parser_t.add_argument("--temp", type=float, default=0.1,
		help="Temperature in Gumbel-Softmax (0.1)")
	parser_t.add_argument("--depth", type=int, default=0,
		help="Number of extra hidden layers in GMVAE model (0)")
	parser_t.add_argument("--latent", action="store_true",
		help="Save latent space parameters")
	parser_t.add_argument("--split", type=float, default=1.0,
		help="Ratio of training/validation (1.0)")
	parser_t.add_argument("--patience", type=int, default=11,
		help="Patience for validation loss (11)")
	parser_t.add_argument("--overlap", type=int, default=0,
		help="Add overlapping SNPs to each end of a window (0)")
	parser_t.add_argument("--save_models", action="store_true",
		help="Save models")
	parser_t.add_argument("--debug", action="store_true",
		help="Print losses")

	# haplonet admix
	parser_a = subparsers.add_parser("admix")
	parser_a.add_argument("-f", "--filelist",
		help="Filelist with paths to multiple log-likelihood files")
	parser_a.add_argument("-l", "--like",
		help="Path to single log-likelihood file")
	parser_a.add_argument("-K", "--K", type=int,
		help="Number of ancestral components")
	parser_a.add_argument("-i", "--iter", type=int, default=5000,
		help="Maximum number of iterations (5000)")
	parser_a.add_argument("-t", "--threads", type=int, default=1,
		help="Number of threads (1)")
	parser_a.add_argument("-c", "--check", type=int, default=50,
		help="Calculating loglike for every i-th iteration (50)")
	parser_a.add_argument("--check_q", action="store_true",
		help="Check convergence for change in Q matrix")
	parser_a.add_argument("-s", "--seed", type=int, default=0,
		help="Random seed (0)")
	parser_a.add_argument("-o", "--out", default="haplonet.admix",
		help="Output path/name ('haplonet.admix')")
	parser_a.add_argument("--tole", type=float, default=0.1,
		help="Difference in loglike between args.check iterations (0.1)")
	parser_a.add_argument("--tole_q", type=float, default=1e-6,
		help="Tolerance for convergence of Q matrix (1e-6)")
	parser_a.add_argument("--no_accel", action="store_true",
		help="Turn off SqS3 acceleration")

	# haplonet pca
	parser_p = subparsers.add_parser("pca")
	parser_p.add_argument("-f", "--filelist",
		help="Filelist with paths to multiple log-likelihood files")
	parser_p.add_argument("-l", "--like",
		help="Path to single log-likelihood file")
	parser_p.add_argument("-F", "--filter", type=float, default=0.0,
		help="Threshold for haplotype cluster frequency (0.0)")
	parser_p.add_argument("-e", "--n_eig", type=int, default=10,
		help="Number of eigenvectors to extract (10)")
	parser_p.add_argument("-c", "--cov", action="store_true",
		help="Estimate covariance matrix instead of SVD")
	parser_p.add_argument("-t", "--threads", type=int, default=1,
		help="Number of threads (1)")
	parser_p.add_argument("-o", "--out", default="haplonet.pca",
		help="Output path/name ('haplonet.pca')")
	parser_p.add_argument("--dosage", action="store_true",
		help="Perform dosage SVD")
	parser_p.add_argument("--iterative", type=int,
		help="Use iterative probabilistic approach")
	parser_p.add_argument("--freqs", action="store_true",
		help="Save haplotype cluster frequencies")
	parser_p.add_argument("--loadings", action="store_true",
		help="Save loadings of SVD")

	# haplonet fatash
	parser_f = subparsers.add_parser("fatash")
	parser_f.add_argument("-f", "--filelist",
		help="Filelist with paths to multiple log-likelihood files")
	parser_f.add_argument("-l", "--like",
		help="Path to single log-likelihood file")
	parser_f.add_argument("-q", "--prop",
		help="Path to estimated global ancestry proportions")
	parser_f.add_argument("-p", "--freq",
		help="Path to ancestral haplotype cluster frequencies")
	parser_f.add_argument("-t", "--threads", type=int,
		help="Number of threads (1)")
	parser_f.add_argument("-o", "--out", default="haplonet.fatash",
		help="Output path/name ('haplonet.fatash')")
	parser_f.add_argument("--alpha", type=float, default=0.01,
		help="Set initial alpha, rate of transition between states")
	parser_f.add_argument("--alpha_optim", action="store_true",
		help="Optimize individual alphas using SciPy")
	parser_f.add_argument("--alpha_bound", nargs=2, type=float,
		default=[0.001, 0.1], help="Bounds on alpha")
	parser_f.add_argument("--alpha_save", action="store_true",
		help="Save individual alphas")
	parser_f.add_argument("--viterbi", action="store_true",
		help="Compute and save viterbi decoding")

	# haplonet convert
	parser_c = subparsers.add_parser("convert")
	parser_c.add_argument("-v", "--vcf",
		help="Input vcf-file of genotypes")
	parser_c.add_argument("-l", "--length", type=int,
		help="Generate median base positions for defined window lengths")
	parser_c.add_argument("-c", "--chromosome", type=int,
		help="Specify chromosome number to avoid ambiguity")
	parser_c.add_argument("-w", "--windows", action="store_true",
		help="Only save median base positions, no .npy output")
	parser_c.add_argument("-o", "--out", default="input",
		help="Output filepath")

	# Parse arguments
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()

	### Run specified command
	if sys.argv[1] == "train":
		if len(sys.argv) < 3:
			parser_t.print_help()
			sys.exit()
		else:
			from haplonet import haploTrain
			deaf = vars(parser_t.parse_args([]))
			haploTrain.main(args, deaf)
	if sys.argv[1] == "admix":
		if len(sys.argv) < 3:
			parser_a.print_help()
			sys.exit()
		else:
			from haplonet import admixNN
			admixNN.main(args)
	if sys.argv[1] == "pca":
		if len(sys.argv) < 3:
			parser_p.print_help()
			sys.exit()
		else:
			from haplonet import pcaNN
			pcaNN.main(args)
	if sys.argv[1] == "fatash":
		if len(sys.argv) < 3:
			parser_f.print_help()
			sys.exit()
		else:
			from haplonet import fatashHMM
			fatashHMM.main(args)
	if sys.argv[1] == "convert":
		if len(sys.argv) < 3:
			parser_c.print_help()
			sys.exit()
		else:
			from haplonet import convertVCF
			convertVCF.main(args)


##### Define main #####
if __name__ == "__main__":
	main()
