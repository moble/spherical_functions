#!/usr/bin/env python

# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from distutils.core import setup

setup(name='spherical_functions',
      version='1.0',
      description='Python/numba implementation of Wigner D Matrices, spin-weighted spherical harmonics, and associated functions',
      author='Michael Boyle',
      # author_email='',
      url='https://github.com/moble/spherical_functions',
      packages=['spherical_functions',],
      package_dir={'spherical_functions': ''},
      package_data={'spherical_functions': ['Wigner_coefficients.npy',
                                            'binomial_coefficients.npy',
                                            'ladder_operator_coefficients.npy']},
     )
