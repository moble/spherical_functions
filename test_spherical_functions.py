#!/usr/bin/env python

from __future__ import print_function, division, absolute_import


# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import warnings
import sys
import math
import cmath
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
        assert np.allclose( sp.WignerD(R, LMpM), sp.WignerD(-R, LMpM), atol=0.0, rtol=precision_WignerD)

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
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # D_{mp,m}(R) = (-1)^{mp+m} \bar{D}_{-mp,-m}(R)
    MpPM = np.array([mp+m for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    LmMpmM = np.array([[ell,-mp,-m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sp.WignerD(R, LMpM), (-1)**MpPM*np.conjugate(sp.WignerD(R, LmMpmM)), atol=0.0, rtol=precision_WignerD)
    # D is a unitary matrix, so its conjugate transpose is its
    # inverse.  D(R) should equal the matrix inverse of D(R^{-1}).
    # So: D_{mp,m}(R) = \bar{D}_{m,mp}(\bar{R})
    LMMp = np.array([[ell,m,mp] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sp.WignerD(R, LMpM), np.conjugate(sp.WignerD(R.conjugate(), LMMp)), atol=0.0, rtol=precision_WignerD)

def test_WignerD_inverse(Rs, ell_max):
    # Ensure that the matrix of the inverse rotation is the inverse of
    # the matrix of the rotation
    for R in Rs:
        for ell in range(ell_max+1):
            LMpM = np.array([[ell,mp,m] for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
            D1 = sp.WignerD(R, LMpM).reshape((2*ell+1,2*ell+1))
            D2 = sp.WignerD(R.inverse(), LMpM).reshape((2*ell+1,2*ell+1))
            assert np.allclose(D1.dot(D2), np.identity(2*ell+1), atol=precision_WignerD, rtol=precision_WignerD)

def test_WignerD_roundoff(Rs,ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Test rotations with |Ra|<1e-15
    expected = [((-1)**ell if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.x, LMpM), expected, atol=0.0, rtol=precision_WignerD)
    expected = [((-1)**(ell+m) if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.y, LMpM), expected, atol=0.0, rtol=precision_WignerD)
    for theta in np.linspace(0,2*np.pi):
        expected = [((-1)**(ell+m) * (np.cos(theta)+1j*np.sin(theta))**(2*m) if mp==-m else 0.0)
                    for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
        assert np.allclose( sp.WignerD(np.cos(theta)*quaternion.y+np.sin(theta)*quaternion.x, LMpM), expected,
                            atol=0.0, rtol=precision_WignerD)
    # Test rotations with |Rb|<1e-15
    expected = [(1.0 if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.one, LMpM), expected, atol=0.0, rtol=precision_WignerD)
    expected = [((-1)**m if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.z, LMpM), expected, atol=0.0, rtol=precision_WignerD)
    for theta in np.linspace(0,2*np.pi):
        expected = [((np.cos(theta)+1j*np.sin(theta))**(2*m) if mp==m else 0.0)
                    for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
        assert np.allclose( sp.WignerD(np.cos(theta)*quaternion.one+np.sin(theta)*quaternion.z, LMpM), expected,
                            atol=0.0, rtol=precision_WignerD)

@pytest.mark.xfail
def test_WignerD_underflow(Rs,ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Some possible tests here:

    # For the cases where I expect underflow to occur, ensure that the
    # results are close to those where the small components are
    # exactly zero.

    # For all cases, test that the results are within some large
    # tolerance (1e-8?) of the results where small components are
    # exactly zero.

    # Test rotations with |Ra|~1e-10 and large ell
    assert False
    # Test rotations with |Rb|~1e-10 and large ell
    assert False


@pytest.fixture
def special_angles():
    return np.arange(-1*np.pi, 1*np.pi+0.1, np.pi/4.)
def UglyWignerd(beta, ell, mp, m):
    Prefactor = math.sqrt(math.factorial(ell+mp)*math.factorial(ell-mp)*math.factorial(ell+m)*math.factorial(ell-m))
    s_min = max(0,m-mp)
    s_max = min(ell+m, ell-mp)
    return Prefactor * sum([((-1)**(mp-m+s) * math.cos(beta/2.)**(2*ell+m-mp-2*s) * math.sin(beta/2.)**(mp-m+2*s)
                             / float(math.factorial(ell+m-s)*math.factorial(s)*math.factorial(mp-m+s)*math.factorial(ell-mp-s)))
                            for s in range(s_min,s_max+1)])
def UglyWignerD(alpha, beta, gamma, ell, mp, m):
    return cmath.exp(-1j*mp*alpha)*UglyWignerd(beta, ell, mp, m)*cmath.exp(-1j*m*gamma)
def test_WignerD_values(special_angles, ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Compare with more explicit forms given in Euler angles
    for alpha in special_angles:
        for beta in special_angles:
            for gamma in special_angles:
                assert np.allclose( np.conjugate(np.array([UglyWignerD(alpha, beta, gamma, ell, mp, m) for ell,mp,m in LMpM])),
                                    sp.WignerD(quaternion.from_euler_angles(alpha, beta, gamma), LMpM),
                                    atol=precision_WignerD, rtol=precision_WignerD )

if __name__=='__main__':
    print("This script is intended to be run through py.test")





