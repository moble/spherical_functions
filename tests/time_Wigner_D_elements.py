#! /usr/bin/env ipython

# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import sys
import numpy as np
import random
import quaternion
import numbapro as nb
from numbapro import *
import spherical_functions as sf

from IPython import get_ipython

ipython = get_ipython()

ru = lambda: random.uniform(-1, 1)

q = np.quaternion(ru(), ru(), ru(), ru())
q = q / q.abs()

ells = range(16 + 1)
evals = np.empty_like(ells, dtype=int)
nanoseconds = np.empty_like(ells, dtype=float)
for i, ell_max in enumerate(ells):
    indices = np.array(
        [[ell, mp, m] for ell in range(ell_max + 1) for mp in range(-ell, ell + 1) for m in range(-ell, ell + 1)],
        dtype=int)
    elements = np.empty((indices.shape[0],), dtype=complex)
    result = ipython.magic("timeit -o sf._Wigner_D_element(q.a, q.b, indices, elements)")
    evals[i] = len(indices)
    nanoseconds[i] = 1e9 * result.best / len(indices)
    print("With ell_max={0}, and {1} evaluations, each D component averages {2:.0f} ns".format(ell_max, evals[i],
                                                                                               nanoseconds[i]))
    sys.stdout.flush()
