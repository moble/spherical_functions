#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import warnings
import sys
import os
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


skip_if_fast = pytest.mark.skipif(os.environ.get('FAST'), reason="Takes a long time because of nested loops")

def test_finite_constant_arrays():
    assert np.all(np.isfinite(sp.factorials))
    assert np.all(np.isfinite(sp._binomial_coefficients))
    assert np.all(np.isfinite(sp._ladder_operator_coefficients))
    assert np.all(np.isfinite(sp._Wigner_coefficients))

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

precision_Wigner3j = 1.e-15

def test_Wigner3j():
    assert abs(sp.Wigner3j(2, 6, 4, 0, 0, 0)-0.1869893980016914) < precision_Wigner3j
    ## The following test various symmetries and other properties fo
    ## the Wigner 3-j symbols
    j_max = 8
    for j1 in range(j_max):
        for j2 in range(j_max):
            for j3 in range(j_max):
                # Selection rule
                if((j1+j2+j3)%2!=0):
                    assert abs(sp.Wigner3j(j1,j2,j3,0,0,0)) < precision_Wigner3j
                for m1 in range(-j1,j1+1):
                    for m2 in range(-j2,j2+1):
                        # Selection rule
                        if(abs(j1-j2)>j3 or j1+j2<j3):
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)) < precision_Wigner3j
                        # Test even permutations
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-sp.Wigner3j(j2,j3,j1,m2,-m1-m2,m1)) < precision_Wigner3j
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-sp.Wigner3j(j2,j3,j1,m2,-m1-m2,m1)) < precision_Wigner3j
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-sp.Wigner3j(j3,j1,j2,-m1-m2,m1,m2)) < precision_Wigner3j
                        # Test odd permutations
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-(-1)**(j1+j2+j3)*sp.Wigner3j(j2,j1,j3,m2,m1,-m1-m2)) < precision_Wigner3j
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-(-1)**(j1+j2+j3)*sp.Wigner3j(j1,j3,j2,m1,-m1-m2,m2)) < precision_Wigner3j
                        # Test sign change
                        assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)-(-1)**(j1+j2+j3)*sp.Wigner3j(j1,j2,j3,-m1,-m2,m1+m2)) < precision_Wigner3j
                        # Regge symmetries (skip for non-integer values)
                        if((j2+j3-m1)%2==0) :
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)
                                       -sp.Wigner3j(j1,(j2+j3-m1)//2,(j2+j3+m1)//2,j3-j2,(j2-j3-m1)//2+m1+m2,(j2-j3+m1)//2-m1-m2)) < precision_Wigner3j
                        if((j2+j3-m1)%2==0) :
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)
                                       -sp.Wigner3j(j1,(j2+j3-m1)//2,(j2+j3+m1)//2,j3-j2,(j2-j3-m1)//2+m1+m2,(j2-j3+m1)//2-m1-m2)) < precision_Wigner3j
                        if((j2+j3+m1)%2==0 and (j1+j3+m2)%2==0 and (j1+j2-m1-m2)%2==0):
                            assert abs(sp.Wigner3j(j1,j2,j3,m1,m2,-m1-m2)
                                       -(-1)**(j1+j2+j3)*sp.Wigner3j((j2+j3+m1)//2,(j1+j3+m2)//2,(j1+j2-m1-m2)//2,
                                                                     j1-(j2+j3-m1)//2,j2-(j1+j3-m2)//2,j3-(j1+j2+m1+m2)//2)) < precision_Wigner3j

@pytest.fixture
def Rs():
    ones = [0,-1.,1.]
    rs = [np.quaternion(w,x,y,z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    random.seed(1842)
    rs = rs + [r.normalized() for r in [np.quaternion(random.uniform(-1,1), random.uniform(-1,1),
                                                      random.uniform(-1,1), random.uniform(-1,1) ) for i in range(20)]]
    return np.array(rs)

precision_WignerD = 4.e-14

def test_WignerD_negative_argument(Rs, ell_max):
    # For integer ell, D(R)=D(-R)
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sp.WignerD(R, LMpM), sp.WignerD(-R, LMpM),
                            atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)

@pytest.mark.parametrize("R1", [skip_if_fast(R1) for R1 in Rs()])
def test_WignerD_representation_property(R1,Rs,ell_max):
    # Test the representation property for special and random angles
    # Try half-integers, too
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R2 in Rs:
        D1 = sp.WignerD(R1, LMpM)
        D2 = sp.WignerD(R2, LMpM)
        D12 = np.array([np.sum([D1[sp._Wigner_index(ell,mp,mpp)]*D2[sp._Wigner_index(ell,mpp,m)] for mpp in range(-ell,ell+1)])
                        for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
        assert np.allclose( sp.WignerD(R1*R2, LMpM), D12, atol=ell_max*precision_WignerD)

def test_WignerD_symmetries(Rs, ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # D_{mp,m}(R) = (-1)^{mp+m} \bar{D}_{-mp,-m}(R)
    MpPM = np.array([mp+m for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    LmMpmM = np.array([[ell,-mp,-m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sp.WignerD(R, LMpM), (-1)**MpPM*np.conjugate(sp.WignerD(R, LmMpmM)),
                            atol=ell_max**2*precision_WignerD, rtol=ell_max**2*precision_WignerD)
    # D is a unitary matrix, so its conjugate transpose is its
    # inverse.  D(R) should equal the matrix inverse of D(R^{-1}).
    # So: D_{mp,m}(R) = \bar{D}_{m,mp}(\bar{R})
    LMMp = np.array([[ell,m,mp] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sp.WignerD(R, LMpM), np.conjugate(sp.WignerD(R.inverse(), LMMp)),
                            atol=ell_max**4*precision_WignerD, rtol=ell_max**4*precision_WignerD)

def test_WignerD_inverse(Rs, ell_max):
    # Ensure that the matrix of the inverse rotation is the inverse of
    # the matrix of the rotation
    for R in Rs:
        for ell in range(ell_max+1):
            LMpM = np.array([[ell,mp,m] for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
            D1 = sp.WignerD(R, LMpM).reshape((2*ell+1,2*ell+1))
            D2 = sp.WignerD(R.inverse(), LMpM).reshape((2*ell+1,2*ell+1))
            assert np.allclose(D1.dot(D2), np.identity(2*ell+1),
                               atol=ell_max**4*precision_WignerD, rtol=ell_max**4*precision_WignerD)

def test_WignerD_roundoff(Rs,ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Test rotations with |Ra|<1e-15
    expected = [((-1)**ell if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.x, LMpM), expected,
                        atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)
    expected = [((-1)**(ell+m) if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.y, LMpM), expected,
                        atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)
    for theta in np.linspace(0,2*np.pi):
        expected = [((-1)**(ell+m) * (np.cos(theta)+1j*np.sin(theta))**(2*m) if mp==-m else 0.0)
                    for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
        assert np.allclose( sp.WignerD(np.cos(theta)*quaternion.y+np.sin(theta)*quaternion.x, LMpM), expected,
                            atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)
    # Test rotations with |Rb|<1e-15
    expected = [(1.0 if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.one, LMpM), expected,
                        atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)
    expected = [((-1)**m if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sp.WignerD(quaternion.z, LMpM), expected,
                        atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)
    for theta in np.linspace(0,2*np.pi):
        expected = [((np.cos(theta)+1j*np.sin(theta))**(2*m) if mp==m else 0.0)
                    for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
        assert np.allclose( sp.WignerD(np.cos(theta)*quaternion.one+np.sin(theta)*quaternion.z, LMpM), expected,
                            atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)

def test_WignerD_underflow(Rs,ell_max):
    assert sp.ell_max>=15 # Test can't work if this has been set lower
    # Test |Ra|=1e-10
    R = np.quaternion(1.e-10, 1, 0, 0).normalized()
    print("Note: If this test fails, it might actually be a good thing...")
    for ell in range(ell_max+1):
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                # print(ell,mp,m)
                if abs(mp+m)>32 and (mp+m)>-30:
                    assert sp.WignerD(R, ell, mp, m) == 0j
                elif abs(mp+m)<17 and (mp+m)>-30: # everything else is mixed
                    assert sp.WignerD(R, ell, mp, m) != 0j
    # Test |Rb|=1e-10
    R = np.quaternion(1, 1.e-10, 0, 0).normalized()
    for ell in range(ell_max+1):
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                if abs(m-mp)>32 and (m-mp)>-30:
                    assert sp.WignerD(R, ell, mp, m) == 0j
                elif abs(m-mp)<17 and (m-mp)>-30: # everything else is mixed
                    assert sp.WignerD(R, ell, mp, m) != 0j


def test_WignerD_overflow(Rs,ell_max):
    assert sp.ell_max>=15 # Test can't work if this has been set lower
    # Test |Ra|=1e-10
    R = np.quaternion(1.e-10, 1, 0, 0).normalized()
    print("Note: If this test fails, it might actually be a good thing...")
    for ell in range(ell_max+1):
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                if (mp+m)<-30:
                    assert np.isnan(sp.WignerD(R, ell, mp, m))
                elif (mp+m)>-30: # ==-30 is an edge case
                    assert np.isfinite(sp.WignerD(R, ell, mp, m))
    # Test |Rb|=1e-10
    R = np.quaternion(1, 1.e-10, 0, 0).normalized()
    for ell in range(ell_max+1):
        for mp in range(-ell,ell+1):
            for m in range(-ell,ell+1):
                if (m-mp)<-30:
                    assert np.isnan(sp.WignerD(R, ell, mp, m))
                elif (m-mp)>-30: # ==-30 is an edge case
                    assert np.isfinite(sp.WignerD(R, ell, mp, m))


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
# @pytest.mark.skipif(os.environ.get('FAST'), reason="Takes a long time because of nested loops")
@skip_if_fast
def test_WignerD_values(special_angles, ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Compare with more explicit forms given in Euler angles
    for alpha in special_angles:
        for beta in special_angles:
            for gamma in special_angles:
                assert np.allclose( np.conjugate(np.array([UglyWignerD(alpha, beta, gamma, ell, mp, m) for ell,mp,m in LMpM])),
                                    sp.WignerD(quaternion.from_euler_angles(alpha, beta, gamma), LMpM),
                                    atol=ell_max**6*precision_WignerD, rtol=ell_max**6*precision_WignerD )

precision_SWSH = 2.e-15

def sYlm(ell,s,m,theta,phi):
    "Eq. II.7 of Ajith et al. (2007) 'Data formats...'"
    return (-1)**(-s) * math.sqrt((2*ell+1)/(4*np.pi)) * (-1)**(s+m)*UglyWignerd(theta, ell, -s, m) * cmath.exp(1j*m*phi)
def m2Y22(iota, phi):
    return math.sqrt(5/(64*np.pi)) * (1+math.cos(iota))**2 * cmath.exp(2j*phi)
def m2Y21(iota, phi):
    return math.sqrt(5/(16*np.pi)) * math.sin(iota) * (1+math.cos(iota)) * cmath.exp(1j*phi)
def m2Y20(iota, phi):
    return math.sqrt(15/(32*np.pi)) * math.sin(iota)**2
def m2Y2m1(iota, phi):
    return math.sqrt(5/(16*np.pi)) * math.sin(iota) * (1-math.cos(iota)) * cmath.exp(-1j*phi)
def m2Y2m2(iota, phi):
    return math.sqrt(5/(64*np.pi)) * (1-math.cos(iota))**2 * cmath.exp(-2j*phi)
def test_SWSH_NINJA_values(special_angles, ell_max):
    for iota in special_angles:
        for phi in special_angles:
            assert abs(sYlm(2,-2, 2,iota,phi) - m2Y22(iota,phi)) < ell_max*precision_SWSH
            assert abs(sYlm(2,-2, 1,iota,phi) - m2Y21(iota,phi)) < ell_max*precision_SWSH
            assert abs(sYlm(2,-2, 0,iota,phi) - m2Y20(iota,phi)) < ell_max*precision_SWSH
            assert abs(sYlm(2,-2,-1,iota,phi) - m2Y2m1(iota,phi)) < ell_max*precision_SWSH
            assert abs(sYlm(2,-2,-2,iota,phi) - m2Y2m2(iota,phi)) < ell_max*precision_SWSH
# @pytest.mark.skipif(os.environ.get('FAST'), reason="Takes a long time because of nested loops")
@skip_if_fast
def test_SWSH_values(special_angles, ell_max):
    for iota in special_angles:
        for phi in special_angles:
            for ell in range(ell_max+1):
                for s in range(-ell,ell+1):
                    for m in range(-ell,ell+1):
                        assert abs( (-1)**(-s)*math.sqrt((2*ell+1)/(4*np.pi))
                                    *sp.WignerD(quaternion.from_euler_angles(phi,iota,0),ell,m,-s)
                                    - sYlm(ell,s,m,iota,phi) ) < ell_max**6*precision_SWSH
                        assert abs( (-1)**(s)*math.sqrt((2*ell+1)/(4*np.pi))
                                    *sp.WignerD(quaternion.from_euler_angles(0,iota,phi),ell,s,-m).conjugate()
                                    - sYlm(ell,s,m,iota,phi) ) < ell_max**6*precision_SWSH

if __name__=='__main__':
    print("This script is intended to be run through py.test")





