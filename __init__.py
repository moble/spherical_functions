from __future__ import print_function, division, absolute_import

__all__ = ['wigner3j']

from numpy import array, floor
from math import factorial, sqrt
from numba import njit

## Module constants
ell_max = 32
epsilon = 1.e-14

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


from .wigner3j import wigner3j
