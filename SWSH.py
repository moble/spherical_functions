# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import math
import cmath

import numpy as np
import quaternion
from quaternion.numba_wrapper import njit, jit, int64, xrange

from . import (_Wigner_coefficient as coeff, epsilon, LM_range)


def SWSH(R, s, indices):
    """Spin-weighted spherical harmonic calculation from rotor

    Note that his function is more general than standard Ylm and sYlm functions of angles because it uses quaternion
    rotors instead of angle, and is slightly faster as a result.

    Parameters
    ----------
    R : unit quaternion
        Rotor on which to evaluate the SWSH function.
    s : int
        Spin weight of the field to evaluate
    indices : 2-d array of int
        Array of (ell,m) values to evaluate

    Returns
    -------
    array of complex
        The shape of this array is `indices.shape[0]`, and contains the values of the SWSH for the (ell,m) values
        specified in `indices`.

    """
    values = np.empty((indices.shape[0],), dtype=complex)
    _SWSH(R.a, R.b, s, indices, values)
    return values


def SWSH_grid(R_grid, s, ell_max):
    """Spin-weighted spherical harmonic calculation from rotors representing a grid

    This function is similar to the `SWSH` function, but assumes that the input is an array of rotors, representing
    various points and/or orientations for which to evaluate the SWSH values.  It simply iterates over each of the
    quaternions, and evaluates the separately for each rotor.

    Note that this function takes `ell_max` as its argument, rather than a set of indices.  The output of this
    function is simply the entire range of indices.

    Also note that his function is more general than standard Ylm and sYlm functions of angles because it uses
    quaternion rotors instead of angle, and is slightly faster as a result.

    Parameters
    ----------
    R_grid : array of unit quaternions
        Rotors on which to evaluate the SWSH function.  Note that, for speed, these are assumed to be normalized.
    s : int
        Spin weight of the field to evaluate
    ell_max : int
        Largest ell value in output arrays.  Note that this should probably be `ell_max >= abs(s)`, but the output
        array will contain values corresponding to `ell < abs(s)`.  Those values will be 0.0, but must be present for
        compatibility with `spinsfast`.

    Returns
    -------
    array of complex
        The shape of this array is `R_grid.shape`, with an extra dimension of length N_lm appended.  That extra
        dimension corresponds to the various (ell,m) values in standard order (e.g., as given by LM_range),
        starting from ell=0 for compatibility with spinsfast.

    """
    indices = LM_range(0, ell_max)
    values = np.zeros(R_grid.shape + (indices.shape[0],), dtype=complex)
    it = np.nditer(R_grid, flags=['multi_index'])
    while not it.finished:
        R = it[0][()]
        _SWSH(R.a, R.b, s, indices, values[it.multi_index])
        it.iternext()
    return values


@njit('void(complex128, complex128, int64, int64[:,:], complex128[:])')
def _SWSH(Ra, Rb, s, indices, values):
    """Compute spin-weighted spherical harmonics from rotor components

    This is the core function that does all the work in the
    computation, but it is strict about its inputs, and does not check
    them for validity -- though numba provides some degree of safety.

    _SWSH(Ra, Rb, s, indices, values)

    Parameters
    ----------
    Ra : complex
        Component `a` of the rotor
    Rb : complex
        Component `b` of the rotor
    s : int
        Spin weight of the field to evaluate
    indices : 2-d array of int
        Array of (ell,m) values to evaluate
    values : 1-d array of complex
        Output array to contain values.  Length must equal first dimension of `indices`.  Needed because numba cannot
        create arrays at the moment.

    Returns
    -------
    void
        The input/output array `values` is modified in place.

    """
    N = indices.shape[0]

    # These constants are the recurring quantities in the computation
    # of the matrix values, so we calculate them here just once

    ra, phia = cmath.polar(Ra)
    rb, phib = cmath.polar(Rb)

    if ra <= epsilon:
        for i in xrange(N):
            ell, m = indices[i, 0:2]
            if (m != s or abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                if (ell) % 2 == 0:
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Rb ** (-2 * s)
                else:
                    values[i] = -math.sqrt((2 * ell + 1) / (4 * np.pi)) * Rb ** (-2 * s)

    elif rb <= epsilon:
        for i in xrange(N):
            ell, m = indices[i, 0:2]
            if (m != -s or abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                if (s) % 2 == 0:
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Ra ** (-2 * s)
                else:
                    values[i] = -math.sqrt((2 * ell + 1) / (4 * np.pi)) * Ra ** (-2 * s)

    elif ra < rb:
        # We have to have these two versions (both this ra<rb branch,
        # and ra>=rb below) to avoid overflows and underflows
        absRRatioSquared = -ra * ra / (rb * rb)
        for i in xrange(N):
            ell, m = indices[i, 0:2]
            if (abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                rhoMin = max(0, -m + s)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, ra might be quite
                # small, in which case ra**(-s+m) could be enormous
                # when the exponent (-s+m) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect(
                    coeff(ell, -m, -s) * rb ** (2 * ell + s - m - 2 * rhoMin) * ra ** (-s + m + 2 * rhoMin),
                    phib * (-s - m) + phia * (-s + m))
                if (Prefactor == 0.0j):
                    values[i] = 0.0j
                else:
                    if ((ell + rhoMin) % 2 != 0):
                        Prefactor *= -1
                    rhoMax = min(ell - m, ell + s)
                    N1 = ell - m + 1
                    N2 = ell + s + 1
                    M = -s + m
                    Sum = 1.0
                    for rho in xrange(rhoMax, rhoMin, -1):
                        Sum *= absRRatioSquared * ((N1 - rho) * (N2 - rho)) / (rho * (M + rho))
                        Sum += 1
                    # Sum = 0.0
                    # for rho in xrange(rhoMax, rhoMin-1, -1):
                    # Sum = (  binomial_coefficient(ell-m,rho) * binomial_coefficient(ell+m, ell-rho+s)
                    # + Sum * absRRatioSquared )
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Prefactor * Sum

    else:  # ra >= rb
        # We have to have these two versions (both this ra>=rb branch,
        # and ra<rb above) to avoid overflows and underflows
        absRRatioSquared = -rb * rb / (ra * ra)
        for i in xrange(N):
            ell, m = indices[i, 0:2]
            if (abs(m) > ell or abs(s) > ell):
                values[i] = 0.0j
            else:
                rhoMin = max(0, m + s)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, rb might be quite
                # small, in which case rb**(-s-m) could be enormous
                # when the exponent (-s-m) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect(
                    coeff(ell, m, -s) * ra ** (2 * ell + s + m - 2 * rhoMin) * rb ** (-s - m + 2 * rhoMin),
                    phia * (-s + m) + phib * (-s - m))
                if (Prefactor == 0.0j):
                    values[i] = 0.0j
                else:
                    if ((rhoMin + s) % 2 != 0):
                        Prefactor *= -1
                    rhoMax = min(ell + m, ell + s)
                    N1 = ell + m + 1
                    N2 = ell + s + 1
                    M = -s - m
                    Sum = 1.0
                    for rho in xrange(rhoMax, rhoMin, -1):
                        Sum *= absRRatioSquared * ((N1 - rho) * (N2 - rho)) / (rho * (M + rho))
                        Sum += 1
                    # Sum = 0.0
                    # for rho in xrange(rhoMax, rhoMin-1, -1):
                    # Sum = (  binomial_coefficient(ell+m,rho) * binomial_coefficient(ell-m, ell-rho+s)
                    # + Sum * absRRatioSquared )
                    values[i] = math.sqrt((2 * ell + 1) / (4 * np.pi)) * Prefactor * Sum
