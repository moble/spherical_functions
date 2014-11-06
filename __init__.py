from __future__ import print_function, division, absolute_import

__all__ = ['Wigner3j']

from numpy import array, floor
from math import factorial, sqrt
from sys import float_info

from quaternion.numba_wrapper import njit

## Module constants
ell_max = 32
epsilon = 1.e-14
min_exp = float_info.min_exp
mant_dig = float_info.mant_dig
error_on_bad_indices = True

## Factorial
factorials = array([float(factorial(i)) for i in range(171)])

@njit('f8(i4)')
def factorial(i):
    return factorials[i]


## Binomial coefficients
_binomial_coefficients = array([floor(0.5+factorials[n]/(factorials[k]*factorials[n-k]))
                                for n in range(2*ell_max+1) for k in range(n+1)])

@njit('f8(i4,i4)')
def binomial_coefficient(n,k):
    return _binomial_coefficients[(n*(n+1))//2+k]


## Ladder-operator coefficients
_ladder_operator_coefficients = array([sqrt(ell*(ell+1)-m*(m+1))
                                       for ell in range(ell_max+1) for m in range(-ell,ell+1)])

@njit('f8(i4,i4)')
def ladder_operator_coefficient(ell,m):
    return _ladder_operator_coefficients[ell*(ell+1)+m]


## Coefficients used in constructing the Wigner D matrices
_Wigner_coefficients = array([sqrt( factorials[ell+m]*factorials[ell-m] / (factorials[ell+mp]*factorials[ell-mp] ) )
                              for ell in range(ell_max+1)
                              for mp in range(-ell, ell+1)
                              for m in range(-ell, ell+1) ])

@njit('f8(i4,i4,i4)')
def _Wigner_coefficient(ell,mp,m):
    return _Wigner_coefficients[ell*(ell*(4*ell + 6) + 5)//3 + mp*(2*ell + 1) + m]


from .Wigner3j import Wigner3j
from .WignerD import WignerD, _WignerD
#from .SWSH import SWSH, SpinWeightedSphericalHarmonic, SphericalHarmonic
