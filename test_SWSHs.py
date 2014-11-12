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

precision_SWSH = 2.e-15

def slow_Wignerd(beta, ell, mp, m):
    Prefactor = math.sqrt(math.factorial(ell+mp)*math.factorial(ell-mp)*math.factorial(ell+m)*math.factorial(ell-m))
    s_min = max(0,m-mp)
    s_max = min(ell+m, ell-mp)
    return Prefactor * sum([((-1)**(mp-m+s) * math.cos(beta/2.)**(2*ell+m-mp-2*s) * math.sin(beta/2.)**(mp-m+2*s)
                             / float(math.factorial(ell+m-s)*math.factorial(s)*math.factorial(mp-m+s)*math.factorial(ell-mp-s)))
                            for s in range(s_min,s_max+1)])
def slow_sYlm(ell,s,m,theta,phi):
    "Eq. II.7 of Ajith et al. (2007) 'Data formats...'"
    return (-1)**(-s) * math.sqrt((2*ell+1)/(4*np.pi)) * (-1)**(s+m)*slow_Wignerd(theta, ell, -s, m) * cmath.exp(1j*m*phi)
## This is just to test my implementation of the equations give in the paper
def test_SWSH_NINJA_values(special_angles, ell_max):
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
    for iota in special_angles:
        for phi in special_angles:
            assert abs(slow_sYlm(2,-2, 2,iota,phi) - m2Y22(iota,phi)) < ell_max*precision_SWSH
            assert abs(slow_sYlm(2,-2, 1,iota,phi) - m2Y21(iota,phi)) < ell_max*precision_SWSH
            assert abs(slow_sYlm(2,-2, 0,iota,phi) - m2Y20(iota,phi)) < ell_max*precision_SWSH
            assert abs(slow_sYlm(2,-2,-1,iota,phi) - m2Y2m1(iota,phi)) < ell_max*precision_SWSH
            assert abs(slow_sYlm(2,-2,-2,iota,phi) - m2Y2m2(iota,phi)) < ell_max*precision_SWSH
@slow
def test_SWSH_values(special_angles, ell_max):
    for iota in special_angles:
        for phi in special_angles:
            for ell in range(ell_max+1):
                for s in range(-ell,ell+1):
                    for m in range(-ell,ell+1):
                        R = quaternion.from_euler_angles(phi,iota,0)
                        assert abs( sp.SWSH(R.a, R.b, s, np.array([[ell,m]]))
                                    - slow_sYlm(ell,s,m,iota,phi) ) < ell_max**6 * precision_SWSH

def test_SWSH_WignerD_expression(special_angles, ell_max):
    for iota in special_angles:
        for phi in special_angles:
            for ell in range(ell_max+1):
                for s in range(ell):
                    R = quaternion.from_euler_angles(phi,iota,0)
                    LM = np.array([[ell,m] for m in range(-ell,ell+1)])
                    Y = sp.SWSH(R.a, R.b, s, LM)
                    LMS = np.array([[ell,m,-s] for m in range(-ell,ell+1)])
                    D = np.empty(Y.shape[0], dtype=complex)
                    sp._Wigner_D_element(R.a, R.b, LMS, D)
                    D = (-1)**(s)*math.sqrt((2*ell+1)/(4*np.pi))*D
                    assert np.allclose(Y, D, atol=ell_max**6*precision_SWSH, rtol=ell_max**6*precision_SWSH)


@pytest.mark.xfail
def test_SWSH_spin_behavior(special_angles, ell_max):
    for s in range(4):
        assert False

