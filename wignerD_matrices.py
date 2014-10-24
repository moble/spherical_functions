from __future__ import print_function, division, absolute_import

from . import njit, _wigner_coefficient as coeff, binomial_coefficient, epsilon, min_exp, mant_dig
from math import sqrt, log

_log2 = log(2)

@njit('c16(c16, c16, i4,i4,i4)')
def wignerD(Ra, Rb, ell, mp, m):
    """Calculate the Wigner D matrix element

    Note that no exception is raised if |m|>ell or |mp|>ell, which are
    not valid values of the input indices; instead, the return value
    is simply 0.

    """
    # Set up some constants we'll use
    absRa = abs(Ra)
    absRb = abs(Rb)
    absRRatioSquared = (absRb*absRb/(absRa*absRa) if absRa>epsilon else 0.0)
    absRa_exp = (int(log(absRa)/_log2) if absRa>epsilon else min_exp)
    absRb_exp = (int(log(absRb)/_log2) if absRb>epsilon else min_exp)

    # Check validity of indices
    if(abs(mp)>ell or abs(m)>ell):
        return 0.0+0.0j

    if(absRa<=epsilon or 2*absRa_exp*(mp-m)<min_exp+mant_dig):
        if mp!=m:
            return 0.0j
        else:
            if (ell+mp)%2==0:
                return Rb**(2*m)
            else:
                return -Rb**(2*m)

    if(absRb<=epsilon or 2*absRb_exp*(mp-m)<min_exp+mant_dig):
        if mp!=m:
            return 0.0j
        else:
            return Ra**(2*m)

    rhoMin = max(0,mp-m)
    rhoMax = min(ell+mp,ell-m)
    if(absRa < 1.e-3):
        # In this branch, we deal with NANs in certain cases when
        # absRa is small by separating it out from the main sum,
        Prefactor = coeff(ell, mp, m) * Ra**(m+mp) * Rb**(m-mp)
        absRaSquared = absRa*absRa
        absRbSquared = absRb*absRb
        Sum = 0.0
        for rho in range(rhoMax, rhoMin-1, -1):
            aTerm = absRaSquared**(ell-m-rho);
            if(aTerm != aTerm or aTerm<1.e-100):
                Sum *= absRbSquared
            else:
                if rho%2==0:
                    Sum = ( binomial_coefficient(ell+mp,rho) * binomial_coefficient(ell-mp, ell-rho-m) * aTerm
                            + Sum * absRbSquared )
                else:
                    Sum = ( -binomial_coefficient(ell+mp,rho) * binomial_coefficient(ell-mp, ell-rho-m) * aTerm
                            + Sum * absRbSquared )
        return Prefactor * Sum * absRbSquared**rhoMin
    else:
        Prefactor = coeff(ell, mp, m) * absRa**(2*ell-2*m) * Ra**(m+mp) * Rb**(m-mp)
        Sum = 0.0
        for rho in range(rhoMax, rhoMin-1, -1):
            if rho%2==0:
                Sum = (  binomial_coefficient(ell+mp,rho) * binomial_coefficient(ell-mp, ell-rho-m)
                         + Sum * absRRatioSquared )
            else:
                Sum = ( -binomial_coefficient(ell+mp,rho) * binomial_coefficient(ell-mp, ell-rho-m)
                         + Sum * absRRatioSquared )
        return Prefactor * Sum * absRRatioSquared**rhoMin
