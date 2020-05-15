#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import math
import cmath
import numpy as np
import quaternion
import spherical_functions as sf
import pytest

slow = pytest.mark.slow

eps = np.finfo(float).eps

precision_Wigner_D_element = 4.e-14

def test_complex_powers():
    M = 10_000
    θs = np.concatenate((np.random.rand(100)*6.3,
                         [np.pi * i for i in [0, 1/8, 1/4, 1/3, 1/2, 3/4, 1, 5/4, 3/2, 7/4, 2, 9/4]]))
    zs = np.exp(1j * θs)
    complex_powers(z, M)
