#!/usr/bin/env python

# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import math
import cmath
import numpy as np
import quaternion
import spherical_functions as sf
import pytest

slow = pytest.mark.slow

precision_Wigner_D = 4.e-15

def test_Wigner_D_linear_indices(ell_max):
    for l_min in range(ell_max):
        for l_max in range(l_min+1,ell_max+1):
            LMpM = sf.LMpM_range(l_min, l_max)

            assert len(LMpM)==sf._total_size_D_matrices(l_min, l_max)

            for ell in range(l_min, l_max+1):
                assert list(LMpM[sf._linear_matrix_offset(ell,l_min)]) == [ell,-ell,-ell]

            for ell in range(l_min, l_max+1):
                for mp in range (-ell,ell+1):
                    assert list(LMpM[sf._linear_matrix_diagonal_index(ell,mp) + sf._linear_matrix_offset(ell,l_min)]) == [ell,mp,mp]
                    for m in range(-ell,ell+1):
                        assert list(LMpM[sf._linear_matrix_index(ell,mp,m) + sf._linear_matrix_offset(ell,l_min)]) == [ell,mp,m]

@slow
def test_Wigner_D_matrix(Rs, ell_max):
    for l_min in [0,1,2,ell_max//2,ell_max-1]:
        print("")
        for l_max in range(l_min+1,ell_max+1):
            print("\tWorking on (l_min,l_max)=({0},{1})".format(l_min,l_max))
            LMpM = sf.LMpM_range(l_min, l_max)
            for R in Rs:
                elements = sf.Wigner_D_element(R, LMpM)
                matrix = np.empty(LMpM.shape[0], dtype=complex)
                sf._Wigner_D_matrices(R.a, R.b, l_min, l_max, matrix)
                # print("="*80, "\n", R, "\n\n", LMpM, "\n\n", elements, "\n\n", matrix, "\n\n", elements-matrix, "\n")
                assert np.allclose( elements, matrix,
                                    atol=precision_Wigner_D, rtol=precision_Wigner_D )

def test_Wigner_D_matrices_negative_argument(Rs, ell_max):
    # For integer ell, D(R)=D(-R)
    LMpM = sf.LMpM_range(0, ell_max)
    a = np.empty((LMpM.shape[0],), dtype=complex)
    b = np.empty((LMpM.shape[0],), dtype=complex)
    for R in Rs:
        sf._Wigner_D_matrices(R.a,R.b,0,ell_max,a)
        sf._Wigner_D_matrices(-R.a,-R.b,0,ell_max,b)
        assert np.allclose( a, b, rtol=ell_max*precision_Wigner_D)

@slow
def test_Wigner_D_matrices_representation_property(Rs,ell_max):
    # Test the representation property for special and random angles
    # Try half-integers, too
    LMpM = sf.LMpM_range(0, ell_max)
    print("")
    D1  = np.empty((LMpM.shape[0],), dtype=complex)
    D2  = np.empty((LMpM.shape[0],), dtype=complex)
    D12 = np.empty((LMpM.shape[0],), dtype=complex)
    for i,R1 in enumerate(Rs):
        print("\t{0} of {1}: R1 = {2}".format(i, len(Rs), R1))
        for R2 in Rs:
            R12 = R1*R2
            sf._Wigner_D_matrices(R1.a, R1.b, 0, ell_max, D1)
            sf._Wigner_D_matrices(R2.a, R2.b, 0, ell_max, D2)
            sf._Wigner_D_matrices(R12.a, R12.b, 0, ell_max, D12)
            M12 = np.array([np.sum([D1[sf._Wigner_index(ell,mp,mpp)]*D2[sf._Wigner_index(ell,mpp,m)] for mpp in range(-ell,ell+1)])
                            for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
            assert np.allclose( M12, D12, atol=ell_max*precision_Wigner_D)

def test_Wigner_D_matrices_inverse(Rs, ell_max):
    # Ensure that the matrix of the inverse rotation is the inverse of
    # the matrix of the rotation
    for R in Rs:
        for ell in range(ell_max+1):
            LMpM = sf.LMpM_range(ell, ell)
            D1  = np.empty((LMpM.shape[0],), dtype=complex)
            D2  = np.empty((LMpM.shape[0],), dtype=complex)
            sf._Wigner_D_matrices(R.a, R.b, ell, ell, D1)
            sf._Wigner_D_matrices(R.a.conjugate(), -R.b, ell, ell, D2)
            D1 = D1.reshape((2*ell+1,2*ell+1))
            D2 = D2.reshape((2*ell+1,2*ell+1))
            assert np.allclose(D1.dot(D2), np.identity(2*ell+1),
                               atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)

def test_Wigner_D_element_symmetries(Rs, ell_max):
    LMpM = sf.LMpM_range(0, ell_max)
    # D_{mp,m}(R) = (-1)^{mp+m} \bar{D}_{-mp,-m}(R)
    MpPM = np.array([mp+m for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    LmMpmM = np.array([[ell,-mp,-m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sf.Wigner_D_element(R, LMpM), (-1)**MpPM*np.conjugate(sf.Wigner_D_element(R, LmMpmM)),
                            atol=ell_max**2*precision_Wigner_D, rtol=ell_max**2*precision_Wigner_D)
    # D is a unitary matrix, so its conjugate transpose is its
    # inverse.  D(R) should equal the matrix inverse of D(R^{-1}).
    # So: D_{mp,m}(R) = \bar{D}_{m,mp}(\bar{R})
    LMMp = np.array([[ell,m,mp] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)])
    for R in Rs:
        assert np.allclose( sf.Wigner_D_element(R, LMpM), np.conjugate(sf.Wigner_D_element(R.inverse(), LMMp)),
                            atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)

def test_Wigner_D_element_roundoff(Rs,ell_max):
    LMpM = sf.LMpM_range(0, ell_max)
    # Test rotations with |Ra|<1e-15
    expected = [((-1)**ell if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sf.Wigner_D_element(quaternion.x, LMpM), expected,
                        atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)
    expected = [((-1)**(ell+m) if mp==-m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sf.Wigner_D_element(quaternion.y, LMpM), expected,
                        atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)
    for theta in np.linspace(0,2*np.pi):
        expected = [((-1)**(ell+m) * (np.cos(theta)+1j*np.sin(theta))**(2*m) if mp==-m else 0.0)
                    for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
        assert np.allclose( sf.Wigner_D_element(np.cos(theta)*quaternion.y+np.sin(theta)*quaternion.x, LMpM), expected,
                            atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)
    # Test rotations with |Rb|<1e-15
    expected = [(1.0 if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sf.Wigner_D_element(quaternion.one, LMpM), expected,
                        atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)
    expected = [((-1)**m if mp==m else 0.0) for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
    assert np.allclose( sf.Wigner_D_element(quaternion.z, LMpM), expected,
                        atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)
    for theta in np.linspace(0,2*np.pi):
        expected = [((np.cos(theta)+1j*np.sin(theta))**(2*m) if mp==m else 0.0)
                    for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1)]
        assert np.allclose( sf.Wigner_D_element(np.cos(theta)*quaternion.one+np.sin(theta)*quaternion.z, LMpM), expected,
                            atol=ell_max*precision_Wigner_D, rtol=ell_max*precision_Wigner_D)

@pytest.mark.xfail
def test_Wigner_D_element_underflow(Rs,ell_max):
    #Note: If this test fails, it might actually be a good thing, if things aren't underflowing...
    assert sf.ell_max>=15 # Test can't work if this has been set lower
    eps = 1.e-9
    ell_max = max(sf.ell_max, ell_max)
    # Test |Ra|=1e-10
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(mp+m)>32])
    R = np.quaternion(eps, 1, 0, 0).normalized()
    assert np.all( sf.Wigner_D_element(R, LMpM) == 0j )
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(mp+m)<32])
    R = np.quaternion(eps, 1, 0, 0).normalized()
    assert np.all( sf.Wigner_D_element(R, LMpM) != 0j )
    # Test |Rb|=1e-10
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(m-mp)>32])
    R = np.quaternion(1, eps, 0, 0).normalized()
    assert np.all( sf.Wigner_D_element(R, LMpM) == 0j )
    LMpM = np.array([[ell,mp,m] for ell in range(ell_max+1) for mp in range(-ell,ell+1) for m in range(-ell,ell+1) if abs(m-mp)<32])
    R = np.quaternion(1, eps, 0, 0).normalized()
    assert np.all( sf.Wigner_D_element(R, LMpM) != 0j )

def test_Wigner_D_element_overflow(Rs,ell_max):
    assert sf.ell_max>=15 # Test can't work if this has been set lower
    ell_max = max(sf.ell_max, ell_max)
    LMpM = sf.LMpM_range(0, ell_max)
    # Test |Ra|=1e-10
    R = np.quaternion(1.e-10, 1, 0, 0).normalized()
    assert np.all( np.isfinite( sf.Wigner_D_element(R, LMpM) ) )
    # Test |Rb|=1e-10
    R = np.quaternion(1, 1.e-10, 0, 0).normalized()
    assert np.all( np.isfinite( sf.Wigner_D_element(R, LMpM) ) )


def slow_Wignerd(beta, ell, mp, m):
    # https://en.wikipedia.org/wiki/Wigner_D-matrix#Wigner_.28small.29_d-matrix
    Prefactor = math.sqrt(math.factorial(ell+mp)*math.factorial(ell-mp)*math.factorial(ell+m)*math.factorial(ell-m))
    s_min = max(0,m-mp)
    s_max = min(ell+m, ell-mp)
    return Prefactor * sum([((-1)**(mp-m+s) * math.cos(beta/2.)**(2*ell+m-mp-2*s) * math.sin(beta/2.)**(mp-m+2*s)
                             / float(math.factorial(ell+m-s)*math.factorial(s)*math.factorial(mp-m+s)*math.factorial(ell-mp-s)))
                            for s in range(s_min,s_max+1)])
def slow_Wigner_D_element(alpha, beta, gamma, ell, mp, m):
    # https://en.wikipedia.org/wiki/Wigner_D-matrix#Definition_of_the_Wigner_D-matrix
    return cmath.exp(-1j*mp*alpha)*slow_Wignerd(beta, ell, mp, m)*cmath.exp(-1j*m*gamma)
@slow
def test_Wigner_D_element_values(special_angles, ell_max):
    LMpM = sf.LMpM_range(0, ell_max)
    # Compare with more explicit forms given in Euler angles
    print("")
    for alpha in special_angles:
        print("\talpha={0}".format(alpha)) # Need to show some progress to Travis
        for beta in special_angles:
            for gamma in special_angles:
                # print("\t\tbeta,gamma={0},{1}".format(beta,gamma))
                assert np.allclose( np.conjugate(np.array([slow_Wigner_D_element(alpha, beta, gamma, ell, mp, m) for ell,mp,m in LMpM])),
                                    sf.Wigner_D_element(quaternion.from_euler_angles(alpha, beta, gamma), LMpM),
                                    atol=2*ell_max*precision_Wigner_D, rtol=2*ell_max*precision_Wigner_D )
