#!/usr/bin/env python

# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

# Try to keep imports to a minimum; from the standard library as much
# as possible.  We have to conda install all dependencies, and it's
# not right to make Travis do too much work.
import math
import numpy as np
import quaternion
import spherical_functions as sf
import pytest

import numba  # This is to check to make sure we're actually using numba


def test_constant_as_ell_0_mode(special_angles):
    indices = np.array([[0, 0]])
    np.random.seed(123)
    for imaginary_part in [0.0, 1.0j]:  # Test both real and imaginary constants
        for rep in range(1000):
            constant = np.random.uniform(-1, 1) + imaginary_part * np.random.uniform(-1, 1)
            const_ell_m = sf.constant_as_ell_0_mode(constant)
            for theta in special_angles:
                for phi in special_angles:
                    dot = np.dot(const_ell_m, sf.SWSH(quaternion.from_spherical_coords(theta, phi), 0, indices))
                    assert abs(constant - dot) < 1e-15


def test_vector_as_ell_1_modes(special_angles):
    indices = np.array([[1, -1], [1, 0], [1, 1]])

    def nhat(theta, phi):
        return np.array([math.sin(theta) * math.cos(phi),
                         math.sin(theta) * math.sin(phi),
                         math.cos(theta)])

    np.random.seed(123)
    for rep in range(1000):
        vector = np.random.uniform(-1, 1, size=(3,))
        vec_ell_m = sf.vector_as_ell_1_modes(vector)
        for theta in special_angles:
            for phi in special_angles:
                dot1 = np.dot(vector, nhat(theta, phi))
                dot2 = np.dot(vec_ell_m, sf.SWSH(quaternion.from_spherical_coords(theta, phi), 0, indices)).real
                assert abs(dot1 - dot2) < 1e-15


def test_finite_constant_arrays():
    assert np.all(np.isfinite(sf.factorials))
    assert np.all(np.isfinite(sf._binomial_coefficients))
    assert np.all(np.isfinite(sf._ladder_operator_coefficients))
    assert np.all(np.isfinite(sf._Wigner_coefficients))


def nCk(n, k):
    """Simple binomial function, so we don't have to import anything"""
    from operator import mul  # or mul=lambda x,y:x*y
    from fractions import Fraction
    from functools import reduce

    return int(reduce(mul, (Fraction(n - i, i + 1) for i in range(k)), 1))


def test_factorials():
    for i in range(len(sf.factorials)):
        assert sf.factorial(i) == sf.factorials[i]
        assert float(math.factorial(i)) == sf.factorial(i)


def test_binomial_coefficients():
    for n in range(2 * sf.ell_max + 1):
        for k in range(n + 1):
            a = nCk(n, k)
            b = sf.binomial_coefficient(n, k)
            assert abs(a - b) / (abs(a) + abs(b)) < 3.e-14


def test_ladder_operator_coefficient():
    for ell in range(sf.ell_max + 1):
        for m in range(-ell, ell + 1):
            a = math.sqrt(ell * (ell + 1) - m * (m + 1))
            b = sf.ladder_operator_coefficient(ell, m)
            if (m == ell):
                assert b == 0.0
            else:
                assert abs(a - b) / (abs(a) + abs(b)) < 3e-16


def test_LM_range(ell_max):
    for l_max in range(ell_max + 1):
        for l_min in range(l_max + 1):
            assert np.array_equal(sf.LM_range(l_min, l_max),
                                  np.array([[ell, m] for ell in range(l_min, l_max + 1) for m in range(-ell, ell + 1)]))


def test_LM_index(ell_max):
    for ell_min in range(ell_max + 1):
        LM = sf.LM_range(ell_min, ell_max)
        for ell in range(ell_min, ell_max + 1):
            for m in range(-ell, ell + 1):
                assert np.array_equal(np.array([ell, m]), LM[sf.LM_index(ell, m, ell_min)])


def test_LM_total_size(ell_max):
    for l_min in range(ell_max + 1):
        for l_max in range(l_min, ell_max + 1):
            assert sf.LM_index(l_max + 1, -(l_max + 1), l_min) == sf.LM_total_size(l_min, l_max)


def test_LMpM_range(ell_max):
    for l_max in range(ell_max + 1):
        assert np.array_equal(sf.LMpM_range(l_max, l_max),
                              np.array([[l_max, mp, m]
                                        for mp in range(-l_max, l_max + 1)
                                        for m in range(-l_max, l_max + 1)]))
        for l_min in range(l_max + 1):
            assert np.array_equal(sf.LMpM_range(l_min, l_max),
                                  np.array([[ell, mp, m]
                                            for ell in range(l_min, l_max + 1)
                                            for mp in range(-ell, ell + 1)
                                            for m in range(-ell, ell + 1)]))


def test_LMpM_index(ell_max):
    for ell_min in range(ell_max + 1):
        LMpM = sf.LMpM_range(ell_min, ell_max)
        for ell in range(ell_min, ell_max + 1):
            for mp in range(-ell, ell + 1):
                for m in range(-ell, ell + 1):
                    assert np.array_equal(np.array([ell, mp, m]), LMpM[sf.LMpM_index(ell, mp, m, ell_min)])


def test_LMpM_total_size(ell_max):
    for l_min in range(ell_max + 1):
        for l_max in range(l_min, ell_max + 1):
            assert sf.LMpM_index(l_max + 1, -(l_max + 1), -(l_max + 1), l_min) == sf.LMpM_total_size(l_min, l_max)

