from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension(
				"haplonet.shared_cy",
				["haplonet/shared_cy.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()]
			),
			Extension(
				"haplonet.lahmm_cy",
				["haplonet/lahmm_cy.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()]
			)]

setup(
	name="HaploNet",
	version="0.3",
	description="Gaussian Mixture Variational Autoencoder for Genetic Data",
	author="Jonas Meisner",
	packages=["haplonet"],
	entry_points={
		"console_scripts": ["haplonet=haplonet.haploNet:main"]
	},
	python_requires=">=3.6",
	ext_modules=cythonize(extensions),
	include_dirs=[numpy.get_include()]
)
