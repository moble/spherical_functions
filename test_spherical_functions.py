#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

import numpy as np
import math
import scipy.misc
import spherical_functions as sp
import warnings
import sys
import pytest

special_angles = np.arange(-8*np.pi, 8*np.pi+0.1, np.pi/4.)

@pytest.fixture
def special_orientations():
    return np.array([[alpha,beta,gamma] for gamma in special_angles for beta in special_angles for alpha in special_angles])


def test_factorials():
    for i in range(len(sp.factorials)):
        assert sp.factorial(i) == sp.factorials[i]
        assert float(math.factorial(i)) == sp.factorial(i)

def test_binomial_coefficients():
    for n in range(2*sp.ell_max+1):
        for k in range(n+1):
            a = scipy.misc.comb(n,k)
            b = sp.binomial_coefficient(n,k)
            assert abs(a-b) / (abs(a)+abs(b)) < 3.e-14

def test_ladder_operator_coefficient():
    for ell in range(sp.ell_max+1):
        for m in range(-ell,ell+1):
            a = math.sqrt(ell*(ell+1)-m*(m+1))
            b = sp.ladder_operator_coefficient(ell,m)
            assert a==b

def test_wigner3j():
    pass


if __name__=='__main__':
    print("This script is intended to be run through py.test")
