# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import math
import cmath
import numpy as np
import quaternion
from . import (binomial_coefficient, Delta,
               epsilon, error_on_bad_indices)
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

    else:
        r__2 = complex(ra,-rb)**2
        r__m2 = complex(ra,rb)**2
        for i in xrange(N):
            ell,m = indices[i,0:2]
            if(abs(m)>ell or abs(s)>ell):
                values[i] = 0.0j
            else:
                Pos = 0.0+0.0j
                Neg = 0.0+0.0j
                if ((m-s)%2)==0:
                    for mpp in xrange(ell,0,-1):
                        # Note that the second Delta takes pi/2 as its argument, so
                        # we just take the conjugate transpose.  And since Delta is
                        # alsways real, it's just the transpose.  This is also good
                        # because it minimizes the jumping around when indexing the
                        # array.
                        Constant = Delta(ell,m,mpp)*Delta(ell,-s,mpp)
                        Pos = r__2*Pos + Constant
                        Neg = r__m2*Neg + Constant
                else:
                    for mpp in xrange(ell,0,-1):
                        # Note that the second Delta takes pi/2 as its argument, so
                        # we just take the conjugate transpose.  And since Delta is
                        # alsways real, it's just the transpose.  This is also good
                        # because it minimizes the jumping around when indexing the
                        # array.
                        Constant = Delta(ell,m,mpp)*Delta(ell,-s,mpp)
                        Pos = r__2*Pos + Constant
                        Neg = r__m2*Neg - Constant
                Sum = Pos*r__2 + Delta(ell,m,0)*Delta(ell,-s,0) + Neg*r__m2
                values[i] = math.sqrt((2*ell+1)/(4*np.pi)) * cmath.exp(1j*(phia*(m-s) + phib*(-s-m) + (s-m)*np.pi/2)) * Sum
