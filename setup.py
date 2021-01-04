from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension(
				"admixNN_cy",
				["admixNN_cy.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()]
			),
			Extension(
				"computeDist_cy",
				["computeDist_cy.pyx"],
				extra_compile_args=['-fopenmp', '-g0'],
				extra_link_args=['-fopenmp'],
				include_dirs=[numpy.get_include()],
			)]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)