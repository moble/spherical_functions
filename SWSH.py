from __future__ import print_function, division, absolute_import

from . import njit, jit
import numpy as np
import quaternion


@njit('void(complex128, complex128, int64[:,:], complex128[:])')
def _SWSH(Ra, Rb, indices):
    """Main work function for computing spin-weighted spherical harmonics

    This is the core function that does all the work in the
    computation, but it is strict about its inputs, and does not check
    them for validity -- though numba provides some degree of safety.

    Input arguments
    ===============
    _SWSH(Ra, Rb, indices, elements)

      * Ra, Rb are the complex components of the rotor
      * indices is an array of integers [ell,mp,-s]
      * elements is an array of complex with length equal to the first
        dimension of indices

    The `elements` variable is needed because numba cannot create
    arrays at the moment, but this is modified in place.

    """

    return ( (-1)**(-s)*math.sqrt((2*ell+1)/(4*np.pi)) * sp._WignerD(Ra, Rb, ell, m, -s) )
