#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import math
import numpy as np
import spherical_functions as sp
import pytest


import numba # This is to check to make sure we're actually using numba

def test_finite_constant_arrays():
    assert np.all(np.isfinite(sp.factorials))
    assert np.all(np.isfinite(sp._binomial_coefficients))
    assert np.all(np.isfinite(sp._ladder_operator_coefficients))
    assert np.all(np.isfinite(sp._Wigner_coefficients))

def nCk(n,k):
    """Simple binomial function"""
    from operator import mul # or mul=lambda x,y:x*y
    from fractions import Fraction
    from functools import reduce
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def test_factorials():
    for i in range(len(sp.factorials)):
        assert sp.factorial(i) == sp.factorials[i]
        assert float(math.factorial(i)) == sp.factorial(i)

def test_binomial_coefficients():
    for n in range(2*sp.ell_max+1):
        for k in range(n+1):
            a = nCk(n,k)
            b = sp.binomial_coefficient(n,k)
            assert abs(a-b) / (abs(a)+abs(b)) < 3.e-14

def test_ladder_operator_coefficient():
    for ell in range(sp.ell_max+1):
        for m in range(-ell,ell+1):
            a = math.sqrt(ell*(ell+1)-m*(m+1))
            b = sp.ladder_operator_coefficient(ell,m)
            assert a==b
