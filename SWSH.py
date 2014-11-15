# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import math
import cmath
import numpy as np
import quaternion
from . import (_Wigner_coefficient as coeff, binomial_coefficient,
               epsilon, min_exp, mant_dig, error_on_bad_indices)
from quaternion.numba_wrapper import njit, jit, int64, xrange


def SWSH(R, s, indices):
    """Spin-weighted spherical harmonic calculation from rotor

     * `Ra`, `Rb` are the complex components of the rotor
     * `s` is the spin-weight of the spherical harmonics
     * `indices` is an array of integers [ell,m]

    Note that the functions `Ylm` and `sYlm` give the more usual
    functions of spherical coordinates (theta, phi).  This function is
    more general and slightly faster.

    """
    values = np.empty((indices.shape[0],), dtype=complex)
    _SWSH(R.a, R.b, s, indices, values)
    return values

@njit('void(complex128, complex128, int64, int64[:,:], complex128[:])')
def _SWSH(Ra, Rb, s, indices, values):
    """Compute spin-weighted spherical harmonics from rotor components

    This is the core function that does all the work in the
    computation, but it is strict about its inputs, and does not check
    them for validity -- though numba provides some degree of safety.

    Input arguments
    ===============
    _SWSH(Ra, Rb, indices, values)

      * `Ra`, `Rb`: complex components of the rotor
      * `s`: spin weight
      * `indices`: array of integers [ell,m]
      * `values`: array of complex with length equal to the first
        dimension of `indices`

    The `values` variable is needed because numba cannot create arrays
    at the moment, but this is modified in place.

    """
    N = indices.shape[0]

    # These constants are the recurring quantities in the computation
    # of the matrix values, so we calculate them here just once

    ra,phia = cmath.polar(Ra)
    rb,phib = cmath.polar(Rb)

    if(ra<=epsilon):
        for i in xrange(N):
            ell,m = indices[i,0:2]
            if(m!=s or abs(m)>ell or abs(s)>ell):
                values[i] = 0.0j
            else:
                if (ell)%2==0:
                    values[i] = math.sqrt((2*ell+1)/(4*np.pi))*Rb**(-2*s)
                else:
                    values[i] = -math.sqrt((2*ell+1)/(4*np.pi))*Rb**(-2*s)

    elif(rb<=epsilon):
        for i in xrange(N):
            ell,m = indices[i,0:2]
            if(m!=-s or abs(m)>ell or abs(s)>ell):
                values[i] = 0.0j
            else:
                if (s)%2==0:
                    values[i] = math.sqrt((2*ell+1)/(4*np.pi))*Ra**(-2*s)
                else:
                    values[i] = -math.sqrt((2*ell+1)/(4*np.pi))*Ra**(-2*s)

    elif(ra<rb):
        # We have to have these two versions (both this ra<rb branch,
        # and ra>=rb below) to avoid overflows and underflows
        absRRatioSquared = -ra*ra/(rb*rb)
        for i in xrange(N):
            ell,m = indices[i,0:2]
            if(abs(m)>ell or abs(s)>ell):
                values[i] = 0.0j
            else:
                rhoMin = max(0,-m+s)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, ra might be quite
                # small, in which case ra**(-s+m) could be enormous
                # when the exponent (-s+m) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect( coeff(ell, m, -s) * rb**(2*ell+s-m-2*rhoMin) * ra**(-s+m+2*rhoMin),
                                        phib*(-s-m) + phia*(-s+m) )
                if(Prefactor==0.0j):
                    values[i] = 0.0j
                else:
                    if((ell+rhoMin)%2!=0):
                        Prefactor *= -1
                    rhoMax = min(ell-m,ell+s)
                    Sum = 0.0
                    for rho in xrange(rhoMax, rhoMin-1, -1):
                        Sum = (  binomial_coefficient(ell-m,rho) * binomial_coefficient(ell+m, ell-rho+s)
                                 + Sum * absRRatioSquared )
                    values[i] = math.sqrt((2*ell+1)/(4*np.pi)) * Prefactor * Sum

    else: # ra >= rb
        # We have to have these two versions (both this ra>=rb branch,
        # and ra<rb above) to avoid overflows and underflows
        absRRatioSquared = -rb*rb/(ra*ra)
        for i in xrange(N):
            ell,m = indices[i,0:2]
            if(abs(m)>ell or abs(s)>ell):
                values[i] = 0.0j
            else:
                rhoMin = max(0,m+s)
                # Protect against overflow by decomposing Ra,Rb as
                # abs,angle components and pulling out the factor of
                # absRRatioSquared**rhoMin.  Here, rb might be quite
                # small, in which case rb**(-s-m) could be enormous
                # when the exponent (-s-m) is very negative; adding
                # 2*rhoMin to the exponent ensures that it is always
                # positive, which protects from overflow.  Meanwhile,
                # underflow just goes to zero, which is fine since
                # nothing else should be very large.
                Prefactor = cmath.rect( coeff(ell, m, -s) * ra**(2*ell+s+m-2*rhoMin) * rb**(-s-m+2*rhoMin),
                                        phia*(-s+m) + phib*(-s-m) )
                if(Prefactor==0.0j):
                    values[i] = 0.0j
                else:
                    if((rhoMin+s)%2!=0):
                        Prefactor *= -1
                    rhoMax = min(ell+m,ell+s)
                    Sum = 0.0
                    for rho in xrange(rhoMax, rhoMin-1, -1):
                        Sum = (  binomial_coefficient(ell+m,rho) * binomial_coefficient(ell-m, ell-rho+s)
                                 + Sum * absRRatioSquared )
                    values[i] = math.sqrt((2*ell+1)/(4*np.pi)) * Prefactor * Sum
