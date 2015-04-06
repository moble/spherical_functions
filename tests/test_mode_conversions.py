# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/waveforms/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
from numpy import *
import pytest
import spherical_functions as sf


def test_ethbar_inverse_NP():
    for s in range(-5, 5+1):
        f = np.random.random(81) + 1j * np.random.random(81)
        f[:sf.LM_index(abs(s), -abs(s), 0)] = 0.0

        # Test ethbar_inverse_NP(ethbar_NP(x)) == x
        ethbarf = sf.ethbar_NP(f, s, 0)
        ethbarf_prime = sf.ethbar_inverse_NP(ethbarf, s-1, 0)
        assert np.allclose(f[sf.LM_index(abs(s-1), -abs(s-1), 0):],
                           ethbarf_prime[sf.LM_index(abs(s-1), -abs(s-1), 0):],
                           atol=0, rtol=1e-15)
        if sf.LM_index(abs(s), -abs(s), 0) < sf.LM_index(abs(s-1), -abs(s-1), 0):
            assert abs(ethbarf_prime[sf.LM_index(abs(s), -abs(s), 0):sf.LM_index(abs(s-1), -abs(s-1), 0)]).max() == 0.0

        # Test ethbar_NP(ethbar_inverse_NP(x)) == x
        fprime = sf.ethbar_inverse_NP(f, s, 0)
        ethbar_fprime = sf.ethbar_NP(fprime, s+1, 0)
        assert np.allclose(f[sf.LM_index(abs(s+1), -abs(s+1), 0):],
                           ethbar_fprime[sf.LM_index(abs(s+1), -abs(s+1), 0):],
                           atol=0, rtol=1e-15)
        if sf.LM_index(abs(s), -abs(s), 0) < sf.LM_index(abs(s+1), -abs(s+1), 0):
            assert abs(ethbar_fprime[sf.LM_index(abs(s), -abs(s), 0):sf.LM_index(abs(s+1), -abs(s+1), 0)]).max() == 0.0
