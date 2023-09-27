from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"haplonet.shared_cy",
		["haplonet/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		language="c++",
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), 
	Extension(
		"haplonet.lahmm_cy",
		["haplonet/lahmm_cy.pyx"],
		extra_compile_args=['-fopenmp', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="HaploNet",
	version="0.5",
	description="Gaussian Mixture Variational Autoencoder for Genetic Data",
	author="Jonas Meisner",
	packages=["haplonet"],
	entry_points={
		"console_scripts": ["haplonet=haplonet.haploNet:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		"cython",
		"cyvcf2",
		"numpy",
		"scipy",
		"torch"
	],
	ext_modules=cythonize(extensions, compiler_directives={'language_level':'3'}),
	include_dirs=[numpy.get_include()]
)
