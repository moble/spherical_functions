#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import math
import cmath
import numpy as np
import quaternion
import spherical_functions as sp
import pytest

slow = pytest.mark.slow

precision_WignerD = 4.e-14

def test_WignerD_negative_argument(Rs, ell_max):
    # For integer ell, D(R)=D(-R)
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sp.WignerD(R, LMpM), sp.WignerD(-R, LMpM),
                            atol=ell_max*precision_WignerD, rtol=ell_max*precision_WignerD)

@slow
def test_WignerD_representation_property(Rs,ell_max):
    # Test the representation property for special and random angles
    # Try half-integers, too
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    print("")
    for R1 in Rs:
        print("\tR1 = {0}".format(R1))
        for R2 in Rs:
            D1 = sp.WignerD(R1, LMpM)
            D2 = sp.WignerD(R2, LMpM)
            D12 = np.array([np.sum([D1[sp._Wigner_index(ell,mp,mpp)]*D2[sp._Wigner_index(ell,mpp,m)] for mpp in range(-ell,ell+1)])
                            for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
            assert np.allclose( sp.WignerD(R1*R2, LMpM), D12, atol=ell_max*precision_WignerD)

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
    #Note: If this test fails, it might actually be a good thing, if things aren't underflowing...
    assert sp.ell_max>=15 # Test can't work if this has been set lower
    ell_max = max(sp.ell_max, ell_max)
    # Test |Ra|=1e-10
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(mp+m)>32])
    R = np.quaternion(1.e-10, 1, 0, 0).normalized()
    assert np.all( sp.WignerD(R, LMpM) == 0j )
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(mp+m)<32])
    R = np.quaternion(1.e-10, 1, 0, 0).normalized()
    assert np.all( sp.WignerD(R, LMpM) != 0j )
    # Test |Rb|=1e-10
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(m-mp)>32])
    R = np.quaternion(1, 1.e-10, 0, 0).normalized()
    assert np.all( sp.WignerD(R, LMpM) == 0j )
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(m-mp)<32])
    R = np.quaternion(1, 1.e-10, 0, 0).normalized()
    assert np.all( sp.WignerD(R, LMpM) != 0j )

def test_WignerD_overflow(Rs,ell_max):
    assert sp.ell_max>=15 # Test can't work if this has been set lower
    ell_max = max(sp.ell_max, ell_max)
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Test |Ra|=1e-10
    R = np.quaternion(1.e-10, 1, 0, 0).normalized()
    assert np.all( np.isfinite( sp.WignerD(R, LMpM) ) )
    # Test |Rb|=1e-10
    R = np.quaternion(1, 1.e-10, 0, 0).normalized()
    assert np.all( np.isfinite( sp.WignerD(R, LMpM) ) )


def slow_Wignerd(beta, ell, mp, m):
    Prefactor = math.sqrt(math.factorial(ell+mp)*math.factorial(ell-mp)*math.factorial(ell+m)*math.factorial(ell-m))
    s_min = max(0,m-mp)
    s_max = min(ell+m, ell-mp)
    return Prefactor * sum([((-1)**(mp-m+s) * math.cos(beta/2.)**(2*ell+m-mp-2*s) * math.sin(beta/2.)**(mp-m+2*s)
                             / float(math.factorial(ell+m-s)*math.factorial(s)*math.factorial(mp-m+s)*math.factorial(ell-mp-s)))
                            for s in range(s_min,s_max+1)])
def slow_WignerD(alpha, beta, gamma, ell, mp, m):
    return cmath.exp(-1j*mp*alpha)*slow_Wignerd(beta, ell, mp, m)*cmath.exp(-1j*m*gamma)
@slow
def test_WignerD_values(special_angles, ell_max):
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    # Compare with more explicit forms given in Euler angles
    print("")
    for alpha in special_angles:
        print("\talpha={0}".format(alpha)) # Need to show some progress to Travis
        for beta in special_angles:
            for gamma in special_angles:
                assert np.allclose( np.conjugate(np.array([slow_WignerD(alpha, beta, gamma, ell, mp, m) for ell,mp,m in LMpM])),
                                    sp.WignerD(quaternion.from_euler_angles(alpha, beta, gamma), LMpM),
                                    atol=ell_max**6*precision_WignerD, rtol=ell_max**6*precision_WignerD )
