#!/usr/bin/env python

from __future__ import print_function, division, absolute_import


# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import warnings
import sys
import math
import numpy as np
import quaternion
import spherical_functions as sp
import pytest
from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction
from functools import reduce
import random

import numba # This is to check to make sure we're actually using numba

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
    assert abs(sp.Wigner3j(2, 6, 4, 0, 0, 0)-0.1869893980016914) < 1.e-15
    ## The following test various symmetries and other properties fo
    ## the Wigner 3-j symbols
    j_max = 8
    for j1 in range(j_max):
        for j2 in range(j_max):
            for j3 in range(j_max):
                # Selection rule
                if((j1+j2+j3)%2!=0):
                    assert abs(sp.Wigner3j(j1,j2,j3,0,0,0)) < 1.e-15
                for m1 in range(-j1,j1+1):
                    for m2 in range(-j2,j2+1):
                        # Selection rule
                        if(abs(j1-j2)>j3 or j1+j2<j3):
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)) < 1.e-15
                        # Test even permutations
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-sp.Wigner3j(j2,j3,j1,m2,-m1-m2,m1)) < 1.e-15
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-sp.Wigner3j(j2,j3,j1,m2,-m1-m2,m1)) < 1.e-15
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-sp.Wigner3j(j3,j1,j2,-m1-m2,m1,m2)) < 1.e-15
                        # Test odd permutations
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-(-1)**(j1+j2+j3)*sp.Wigner3j(j2,j1,j3,m2,m1,-m1-m2)) < 1.e-15
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-(-1)**(j1+j2+j3)*sp.Wigner3j(j1,j3,j2,m1,-m1-m2,m2)) < 1.e-15
                        # Test sign change
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-(-1)**(j1+j2+j3)*sp.Wigner3j(j1,j2,j3,-m1,-m2,m1+m2)) < 1.e-15
                        # Regge symmetries (skip for non-integer values)
                        if((j2+j3-m1)%2==0) :
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)
                                       -sp.Wigner3j(j1,(j2+j3-m1)//2,(j2+j3+m1)//2,j3-j2,(j2-j3-m1)//2+m1+m2,(j2-j3+m1)//2-m1-m2)) < 1.e-15
                        if((j2+j3-m1)%2==0) :
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)
                                       -sp.Wigner3j(j1,(j2+j3-m1)//2,(j2+j3+m1)//2,j3-j2,(j2-j3-m1)//2+m1+m2,(j2-j3+m1)//2-m1-m2)) < 1.e-15
                        if((j2+j3+m1)%2==0 and (j1+j3+m2)%2==0 and (j1+j2-m1-m2)%2==0):
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)
                                       -(-1)**(j1+j2+j3)*sp.Wigner3j((j2+j3+m1)//2,(j1+j3+m2)//2,(j1+j2-m1-m2)//2,
                                                                     j1-(j2+j3-m1)//2,j2-(j1+j3-m2)//2,j3-(j1+j2+m1+m2)//2)) < 1.e-15

@pytest.fixture
def Rs():
    ones = [0,-1.,1.]
    rs = [np.quaternion(w,x,y,z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    random.seed(1842)
    rs = rs + [r.normalized() for r in [np.quaternion(random.uniform(-1,1), random.uniform(-1,1),
                                                      random.uniform(-1,1), random.uniform(-1,1) ) for i in range(20)]]
    return np.array(rs)

precision_WignerD = 3.e-14

def test_WignerD_negative_argument(Rs, ell_max):
    # For integer ell, D(R)=D(-R)
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        print(R)
        assert np.allclose( sp.WignerD(R, LMpM), sp.WignerD(-R, LMpM), rtol=precision_WignerD)

def test_WignerD_representation_property(Rs,ell_max):
    # Test the representation property for special and random angles
    # Try half-integers, too
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R1 in Rs:
        for R2 in Rs:
            D1 = sp.WignerD(R1, LMpM)
            D2 = sp.WignerD(R2, LMpM)
            D12 = np.array([np.sum([D1[sp._Wigner_index(ell,mp,mpp)]*D2[sp._Wigner_index(ell,mpp,m)] for mpp in range(-ell,ell+1)])
                            for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
            assert np.allclose( sp.WignerD(R1*R2, LMpM), D12, atol=precision_WignerD)

def test_WignerD_symmetries(Rs, ell_max):
    # Test symmetries
    pass

def test_WignerD_roundoff_cases(Rs,ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Test rotations with |Ra|<1e-15
    expected = [((-1)**ell if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.x, LMpM), expected, rtol=precision_WignerD)
    expected = [((-1)**(ell+m) if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.y, LMpM), expected, rtol=precision_WignerD)
    # Test rotations with |Rb|<1e-15
    expected = [(1.0 if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.one, LMpM), expected, rtol=precision_WignerD)
    expected = [((-1)**m if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.z, LMpM), expected, rtol=precision_WignerD)
    # Test rotations with |Ra|~1e-10 and large ell
    pass
    # Test rotations with |Rb|~1e-10 and large ell
    pass

if __name__=='__main__':
    print("This script is intended to be run through py.test")





