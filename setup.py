#!/usr/bin/env python

from distutils.core import setup

setup(name='spherical_functions',
      version='1.0',
      description='Python/numba implementation of Wigner D Matrices, spin-weighted spherical harmonics, and associated functions',
      author='Michael Boyle',
      # author_email='',
      url='https://github.com/moble/spherical_functions',
      package_dir={'spherical_functions': ''},
      packages=['spherical_functions',],
     )
