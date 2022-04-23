from setuptools import setup
from Cython.Build import cythonize
import numpy as np

include_path = [np.get_include()]
ext_modules = cythonize("*.pyx", include_path = include_path)
for ext in ext_modules:
      ext.include_dirs = include_path

setup(name = 'libvoxelize',
      ext_modules = cythonize("*.pyx"))
