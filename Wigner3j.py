# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

"""Module for computing Wigner's 3-j symbols

This module uses code from sympy, originally written by Jens Rasch.

"""

from __future__ import print_function, division, absolute_import

from math import sqrt
from . import factorials
from quaternion.numba_wrapper import njit, xrange


@njit('f8(i8,i8,i8,i8,i8,i8)')
def Wigner3j(j_1, j_2, j_3, m_1, m_2, m_3):
    """Calculate the Wigner 3j symbol `Wigner3j(j_1,j_2,j_3,m_1,m_2,m_3)`

    This function is copied with minor modification from
    sympy.physics.Wigner, as written by Jens Rasch.

    The inputs must be integers.  (Half integer arguments are
    sacrificed so that we can use numba.)  Nonzero return quantities
    only occur when the `j`s obey the triangle inequality (any two
    must add up to be as big as or bigger than the third).

    Examples
    ========

    >>> from spherical_functions import Wigner3j
    >>> Wigner3j(2, 6, 4, 0, 0, 0)
    0.186989398002
    >>> Wigner3j(2, 6, 4, 0, 0, 1)
    0

    """
    if (m_1 + m_2 + m_3 != 0):
        return 0
    if ( (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3) ):
        return 0
    prefid = (1 if (j_1 - j_2 - m_3) % 2 == 0 else -1)
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if (a1 < 0):
        return 0
    a2 = j_1 - j_2 + j_3;
    if (a2 < 0):
        return 0
    a3 = -j_1 + j_2 + j_3;
    if (a3 < 0):
        return 0

    argsqrt = ( factorials[j_1 + j_2 - j_3] *
                factorials[j_1 - j_2 + j_3] *
                factorials[-j_1 + j_2 + j_3] *
                factorials[j_1 - m_1] *
                factorials[j_1 + m_1] *
                factorials[j_2 - m_2] *
                factorials[j_2 + m_2] *
                factorials[j_3 - m_3] *
                factorials[j_3 + m_3] ) / factorials[j_1 + j_2 + j_3 + 1]

    ressqrt = sqrt(argsqrt)

    imin = max(-j_3 + j_1 + m_2, max(-j_3 + j_2 - m_1, 0))
    imax = min(j_2 + m_2, min(j_1 - m_1, j_1 + j_2 - j_3))
    sumres = 0.0;
    for ii in xrange(imin, imax + 1):
        den = ( factorials[ii] *
                factorials[ii + j_3 - j_1 - m_2] *
                factorials[j_2 + m_2 - ii] *
                factorials[j_1 - ii - m_1] *
                factorials[ii + j_3 - j_2 + m_1] *
                factorials[j_1 + j_2 - j_3 - ii] )
        if (ii % 2 == 0):
            sumres = sumres + 1.0 / den
        else:
            sumres = sumres - 1.0 / den

    return ressqrt * sumres * prefid
