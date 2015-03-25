#!/usr/bin/env python

# Copyright (c) 2015, Michael Boyle
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

precision_Wigner3j = 1.e-15


def test_Wigner3j():
    assert abs(sf.Wigner3j(2, 6, 4, 0, 0, 0) - 0.1869893980016914) < precision_Wigner3j
    ## The following test various symmetries and other properties fo
    ## the Wigner 3-j symbols
    j_max = 8
    for j1 in range(j_max):
        for j2 in range(j_max):
            for j3 in range(j_max):
                # Selection rule
                if ((j1 + j2 + j3) % 2 != 0):
                    assert abs(sf.Wigner3j(j1, j2, j3, 0, 0, 0)) < precision_Wigner3j
                for m1 in range(-j1, j1 + 1):
                    for m2 in range(-j2, j2 + 1):
                        # Selection rule
                        if abs(j1 - j2) > j3 or j1 + j2 < j3:
                            assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)) < precision_Wigner3j
                        # Test even permutations
                        assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                   - sf.Wigner3j(j2, j3, j1, m2, -m1 - m2, m1)) < precision_Wigner3j
                        assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                   - sf.Wigner3j(j2, j3, j1, m2, -m1 - m2, m1)) < precision_Wigner3j
                        assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                   - sf.Wigner3j(j3, j1, j2, -m1 - m2, m1, m2)) < precision_Wigner3j
                        # Test odd permutations
                        assert abs(
                            sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                            - (-1) ** (j1 + j2 + j3) * sf.Wigner3j(j2, j1, j3, m2, m1, -m1 - m2)) < precision_Wigner3j
                        assert abs(
                            sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                            - (-1) ** (j1 + j2 + j3) * sf.Wigner3j(j1, j3, j2, m1, -m1 - m2, m2)) < precision_Wigner3j
                        # Test sign change
                        assert abs(
                            sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                            - (-1) ** (j1 + j2 + j3) * sf.Wigner3j(j1, j2, j3, -m1, -m2, m1 + m2)) < precision_Wigner3j
                        # Regge symmetries (skip for non-integer values)
                        if ((j2 + j3 - m1) % 2 == 0):
                            assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                       - sf.Wigner3j(j1, (j2 + j3 - m1) // 2, (j2 + j3 + m1) // 2, j3 - j2,
                                                     (j2 - j3 - m1) // 2 + m1 + m2,
                                                     (j2 - j3 + m1) // 2 - m1 - m2)) < precision_Wigner3j
                        if ((j2 + j3 - m1) % 2 == 0):
                            assert abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                       - sf.Wigner3j(j1, (j2 + j3 - m1) // 2, (j2 + j3 + m1) // 2, j3 - j2,
                                                     (j2 - j3 - m1) // 2 + m1 + m2,
                                                     (j2 - j3 + m1) // 2 - m1 - m2)) < precision_Wigner3j
                        if ((j2 + j3 + m1) % 2 == 0 and (j1 + j3 + m2) % 2 == 0 and (j1 + j2 - m1 - m2) % 2 == 0):
                            assert (abs(sf.Wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
                                        - (-1) ** (j1 + j2 + j3) * sf.Wigner3j((j2 + j3 + m1) // 2,
                                                                               (j1 + j3 + m2) // 2,
                                                                               (j1 + j2 - m1 - m2) // 2,
                                                                               j1 - (j2 + j3 - m1) // 2,
                                                                               j2 - (j1 + j3 - m2) // 2,
                                                                               j3 - (j1 + j2 + m1 + m2) // 2))
                                    < precision_Wigner3j)
