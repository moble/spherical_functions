from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
from . import (_Wigner_coefficient as coeff, binomial_coefficient,
               epsilon, min_exp, mant_dig, error_on_bad_indices)
from quaternion.numba_wrapper import njit, jit, int64, xrange

epsilon_squared = epsilon*epsilon

_log2 = np.log(2)

@njit('b1(i8,i8,i8)')
def _check_valid_indices(ell, mp, m):
    if(abs(mp)>ell or abs(m)>ell):
        return False
    return True

def WignerD(*args):
    """Return elements of the Wigner D matrices

    The conventions used for this function are discussed more fully on
    <http://moble.github.io/spherical_functions/>.

    Input arguments
    ===============
    The input can be in any of the following forms:

    Wigner(R, ell, mp, m)
    Wigner(R, indices)
    Wigner(Ra, Rb, ell, mp, m)
    Wigner(Ra, Rb, indices)
    Wigner(alpha, beta, gamma, ell, mp, m)
    Wigner(alpha, beta, gamma, indices)

    Where
      * R is a unit quaternion (no checking of norm is done)
      * Ra and Rb are the complex parts of a unit quaternion
      * alpha, beta, gamma are the Euler angles [shudder...]
      * ell, mp, m are the integral indices of the D matrix element
      * indices is an array of [ell,mp,m] indices as above, or simply
        a list of ell modes, in which case all valid [mp,m] values
        will be returned

    Note that there is currently no support for half-integral indices,
    though this would be very simple to implement.  Basically, the
    lack of support is simply due to the fact that the compiled code
    can run faster with integer arguments.  Feel free to open an issue
    on this project's github page if you want support for half-integer
    arguments <https://github.com/moble/spherical_functions/issues>.

    Also note that, by default, a ValueError will be raised if the
    input (ell, mp, m) values are not valid.  (For example, |m|>ell.)
    If instead, you would simply like a return value of 0.0, after
    importing this module as sp, simply evaluate
    >>> sp.error_on_bad_indices = False


    Return value
    ============

    One complex number is returned for each component requested.  If
    the (ell,mp,m) arguments were given explicitly, this means that a
    single complex scalar is returned.  If more than one component was
    requested, a one-dimensional numpy array of complex scalars is
    returned, in the same order as the input.

    """
    # Find the rotation from the args
    if isinstance(args[0], np.quaternion):
        # The rotation is input as a single quaternion
        Ra = args[0].a
        Rb = args[0].b
        mode_offset = 1
    elif isinstance(args[0], complex) and isinstance(args[1], complex):
        # The rotation is input as the two parts of a single quaternion
        Ra = args[0]
        Rb = args[1]
        mode_offset = 2
    elif isinstance(args[0], (int,float)) and isinstance(args[1], (int,float)) and isinstance(args[2], (int,float)):
        # UUUUGGGGLLLLYYYY.  The rotation is input as Euler angles
        R = quaternion.from_Euler_angles(args[0], args[1], args[2])
        Ra = R.a
        Rb = R.b
        mode_offset = 3
    else:
        raise ValueError("Can't understand input rotation")

    # Find the indices
    return_scalar = False
    if(len(args)-mode_offset == 3):
        # Assume these are the (l, mp, m) indices
        ell,mp,m = args[mode_offset:]
        indices = np.array([[ell,mp,m],], dtype=int)
        if(error_on_bad_indices and not _check_valid_indices(*(indices[0]))):
            raise ValueError("(ell,mp,m)=({0},{1},{2}) is not a valid set of indices for Wigner's D matrix".format(ell,mp,m))
        return_scalar = True
    elif(len(args)-mode_offset == 1):
        indices = np.asarray(args[mode_offset], dtype=int)
        if(indices.ndim==0 and indices.size==1):
            # This was just a single ell value
            ell = indices[0]
            indices = np.array([[ell, mp, m] for mp in xrange(-ell,ell+1) for m in xrange(-ell,ell+1)])
            if(ell==0):
                return_scalar = True
        elif(indices.ndim==1 and indices.size>0):
            # This a list of ell values
            indices = np.array([[ell, mp, m] for ell in indices for mp in xrange(-ell,ell+1) for m in xrange(-ell,ell+1)])
        elif(indices.ndim==2):
            # This is an array of [ell,mp,m] values
            if(error_on_bad_indices):
                for ell,mp,m in indices:
                    if not _check_valid_indices(ell,mp,m):
                        raise ValueError("(ell,mp,m)=({0},{1},{2}) is not a valid set of indices for Wigner's D matrix".format(ell,mp,m))
        else:
            raise ValueError("Can't understand input indices")
    else:
        raise ValueError("Can't understand input indices")

    elements = np.empty((len(indices),), dtype=complex)
    _WignerD(Ra, Rb, indices, elements)

    if(return_scalar):
        return elements[0]
    return elements

@njit('void(complex128, complex128, int64[:,:], complex128[:])')
def _WignerD(Ra, Rb, indices, elements):
    """Main work function for computing Wigner D matrix elements

    This is the core function that does all the work in the
    computation, but it is strict about its input, and does not check
    them for validity.

    Input arguments
    ===============
    _WignerD(Ra, Rb, indices, elements)

      * Ra, Rb are the complex components of the rotor
      * indices is an array of integers [ell,mp,m]
      * elements is an array of complex with length equal to the first
        dimension of indices

    The `elements` variable is needed because numba cannot create
    arrays at the moment, but this is modified in place.

    """
    N = indices.shape[0]

    # These constants are the recurring quantities in the computation
    # of the matrix elements, so we calculate them here just once
    absRa_squared = abs(Ra)**2
    absRb_squared = abs(Rb)**2
    absRRatioSquared = (-absRb_squared/absRa_squared if absRa_squared>=absRb_squared else -absRa_squared/absRb_squared)

    if(absRa_squared<=epsilon_squared):
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(mp!=-m or abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                if (ell+m)%2==0:
                    elements[i] = Rb**(2*m)
                else:
                    elements[i] = -Rb**(2*m)

    elif(absRb_squared<=epsilon_squared):
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(mp!=m or abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                elements[i] = Ra**(2*m)

    elif(absRa_squared<absRb_squared):
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                Prefactor = coeff(ell, mp, m) * absRb_squared**(ell-m) * Rb**(m-mp) * Ra**(m+mp)
                if((ell+m)%2!=0):
                    Prefactor *= -1
                if(Prefactor==0.0j):
                    elements[i] = 0.0j
                else:
                    rhoMin = max(0,-mp-m)
                    rhoMax = min(ell-mp,ell-m)
                    Sum = 0.0
                    for rho in xrange(rhoMax, rhoMin-1, -1):
                        Sum = (  binomial_coefficient(ell-mp,rho) * binomial_coefficient(ell+mp, ell-rho-m)
                                 + Sum * absRRatioSquared )
                    elements[i] = Prefactor * Sum * absRRatioSquared**rhoMin

    else: # absRa_squared >= absRb_squared
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                Prefactor = coeff(ell, mp, m) * absRa_squared**(ell-m) * Ra**(m+mp) * Rb**(m-mp)
                if(Prefactor==0.0j):
                    elements[i] = 0.0j
                else:
                    rhoMin = max(0,mp-m)
                    rhoMax = min(ell+mp,ell-m)
                    Sum = 0.0
                    for rho in xrange(rhoMax, rhoMin-1, -1):
                        Sum = (  binomial_coefficient(ell+mp,rho) * binomial_coefficient(ell-mp, ell-rho-m)
                                 + Sum * absRRatioSquared )
                    elements[i] = Prefactor * Sum * absRRatioSquared**rhoMin
