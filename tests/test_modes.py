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

def test_modes_creation():
    s = -2
    ell_min = abs(s)
    ell_max = 8
    a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2)
    with pytest.raises(ValueError):
        m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
    a = a.view(complex)
    m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
    
