#!/usr/bin/env python

# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import sympy
import numpy as np
import quaternion
import spherical_functions as sf
from spherical_functions.WignerDRecursion import HCalculator


def test_WignerDRecursion_accuracy():
    from sympy.physics.quantum.spin import WignerD as sympyWignerD

    """Eq. (29) of arxiv:1403.7698: d^{m',m}_{n}(β) = ϵ(m') ϵ(-m) H^{m',m}_{n}(β)"""

    def ϵ(m):
        m = np.asarray(m)
        eps = np.ones_like(m)
        eps[m >= 0] = (-1)**m[m >= 0]
        return eps

    ell_max = 4
    alpha, beta, gamma = 0.0, 0.1, 0.0
    hcalc = HCalculator(ell_max)
    Hnmpm = hcalc(np.cos(beta))
    max_error = 0.0
    # errors = np.empty(hcalc.nmpm_total_size)

    i = 0
    print()
    for n in range(hcalc.n_max+1):
        print('Testing n={} compared to sympy'.format(n))
        for mp in range(-n, n+1):
            for m in range(-n, n+1):
                sympyd = sympy.re(sympy.N(sympyWignerD(n, mp, m, alpha, -beta, gamma).doit()))
                myd = ϵ(mp) * ϵ(-m) * Hnmpm[i]
                error = float(min(abs(sympyd+myd), abs(sympyd-myd)))
                assert error < 1e-14, "Testing Wigner d recursion: n={}, m'={}, m={}, error={}".format(n, mp, m, error)
                max_error = max(error, max_error)
                # errors[i] = float(min(abs(sympyd+myd), abs(sympyd-myd)))
                # print("{:>5} {:>5} {:>5} {:24} {:24} {:24}".format(n, mp, m, float(sympyd), myd, errors[i]))
                i += 1

    print("Testing Wigner d recursion: max error = {}".format(max_error))


def test_WignerDRecursion_timing():
    import timeit
    import textwrap
    ell_max = 8
    cosβ = 2*np.random.rand(2*ell_max+1, 2*ell_max+1) - 1
    hcalc = HCalculator(ell_max)
    workspace = hcalc.workspace(cosβ)
    time = timeit.timeit('hcalc(cosβ, workspace=workspace)', number=100, globals={'hcalc': hcalc, 'cosβ': cosβ, 'workspace': workspace})
    print('\nTime for ell_max={} grid points was {}ns'.format(ell_max, time))
    cosβ = 2*np.random.rand(2*100+1, 2*100+1) - 1
    workspace = hcalc.workspace(cosβ)
    time = timeit.timeit('hcalc(cosβ, workspace=workspace)', number=100, globals={'hcalc': hcalc, 'cosβ': cosβ, 'workspace': workspace})
    print('Time for ell_max={} grid points was {}ns'.format(100, time))


def test_WignerDRecursion_lineprofiling():
    from line_profiler import LineProfiler
    ell_max = 8
    hcalc = HCalculator(ell_max)
    cosβ = 2*np.random.rand(100, 100) - 1
    workspace = hcalc.workspace(cosβ)
    profiler = LineProfiler(hcalc.__call__)#, hcalc._step_2, hcalc._step_3, hcalc._step_4, hcalc._step_5, hcalc._step_6)
    profiler.runctx('hcalc(cosβ, workspace=workspace)', {'hcalc': hcalc, 'cosβ': cosβ, 'workspace': workspace}, {})
    print()
    profiler.print_stats()