#!/usr/bin/env python

from __future__ import print_function, division, absolute_import


# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import warnings
import sys
import math
import numpy as np
import spherical_functions as sp
import pytest
from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction
from functools import reduce

import numba # This is to check to make sure we're actually using numba

special_angles = np.arange(-8*np.pi, 8*np.pi+0.1, np.pi/4.)

@pytest.fixture
def special_orientations():
    return np.array([[alpha,beta,gamma] for gamma in special_angles for beta in special_angles for alpha in special_angles])

def nCk(n,k):
    """Simple binomial function"""
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

def test_Wigner3j():
    pass
    # assert sp.Wigner3j


if __name__=='__main__':
    print("This script is intended to be run through py.test")
