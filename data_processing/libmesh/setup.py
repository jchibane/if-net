from setuptools import setup
from Cython.Build import cythonize

setup(name = 'libmesh',
      ext_modules = cythonize("*.pyx"))
