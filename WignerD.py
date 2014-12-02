# Copyright (c) 2014, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

"""Module for computing Wigner D functions.

This module contains various functions such as Wigner_D_element and
Wigner_D_matrices (as well as faster variants, designated by underscores,
which assume correct inputs), which calculate the elements and complete
matrices (respectively) of Wigner's D symbols.

"""

from __future__ import print_function, division, absolute_import

import cmath
import numpy as np
import quaternion
from . import (binomial_coefficient, Delta,
               epsilon, error_on_bad_indices, LMpM_total_size,
               ell_max as sf_ell_max)
from quaternion.numba_wrapper import njit, jit, int64, complex128, xrange

_log2 = np.log(2)

@njit('b1(i8,i8,i8)')
def _check_valid_indices(ell, mp, m):
    if(ell>sf_ell_max or abs(mp)>ell or abs(m)>ell):
        return False
    return True

def Wigner_D_element(*args):
    """Return elements of the Wigner D matrices

    The conventions used for this function are discussed more fully on
    <http://moble.github.io/spherical_functions/>.

    Input arguments
    ===============
    The input can be in any of the following forms:

    Wigner_D_element(R, ell, mp, m)
    Wigner_D_element(R, indices)
    Wigner_D_element(Ra, Rb, ell, mp, m)
    Wigner_D_element(Ra, Rb, indices)
    Wigner_D_element(alpha, beta, gamma, ell, mp, m)
    Wigner_D_element(alpha, beta, gamma, indices)

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
    importing this module as sf, simply evaluate
    >>> sf.error_on_bad_indices = False


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
    _Wigner_D_element(Ra, Rb, indices, elements)

    if(return_scalar):
        return elements[0]
    return elements



@njit('void(complex128, complex128, int64[:,:], complex128[:])')
def _Wigner_D_element(Ra, Rb, indices, elements):
    """Main work function for computing Wigner D matrix elements

    This is the core function that does all the work in the
    computation, but it is strict about its input, and does not check
    them for validity.

    Input arguments
    ===============
    _Wigner_D_element(Ra, Rb, indices, elements)

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
    ra,phia = cmath.polar(Ra)
    rb,phib = cmath.polar(Rb)

    if(ra<=epsilon):
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(mp!=-m or abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                if (ell+m)%2==0:
                    elements[i] = Rb**(2*m)
                else:
                    elements[i] = -Rb**(2*m)
    elif(rb<=epsilon):
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(mp!=m or abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                elements[i] = Ra**(2*m)
    else:
        r__2 = complex(ra,-rb)**2
        phi = cmath.phase(r__2)
        for i in xrange(N):
            ell,mp,m = indices[i,0:3]
            if(abs(mp)>ell or abs(m)>ell):
                elements[i] = 0.0j
            else:
                Sum = 0.0
                for mpp in xrange(ell,-ell-1,-1):
                    # Note that the second Delta takes pi/2 as its argument, so
                    # we just take the conjugate transpose.  And since Delta is
                    # alsways real, it's just the transpose.  This is also good
                    # because it minimizes the jumping around when indexing the
                    # array.
                    Sum = (Delta(ell,mp,mpp)*Delta(ell,m,mpp)) + r__2*Sum
                elements[i] = cmath.exp(1j*(phia*(mp+m) + phib*(m-mp) - phi*(ell) + (m-mp)*np.pi/2)) * Sum




@njit('int64(int64, int64, int64)')
def _linear_matrix_index(ell,mp,m):
    """Index of array corresponding to matrix element

    This gives the index based at the first element of the matrix, so
    if the array is actually a series of matrices (linearized), then
    that initial index for this matrix must be added.

    This assumes that the input array corresponds to

    [[ell,mp,m] for ell in range(ell_min,ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    """
    return (ell+m) + (ell+mp)*(2*ell+1)

@njit('int64(int64, int64)')
def _linear_matrix_diagonal_index(ell,mpm):
    """Index of array corresponding to matrix diagonal element

    This gives the index based at the first element of the matrix, so
    if the array is actually a series of matrices (linearized), then
    that initial index for this matrix must be added.

    This assumes that the input array corresponds to

    [[ell,mp,m] for ell in range(ell_min,ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    """
    return (ell+mpm)*(2*ell+2)

@njit('int64(int64, int64)')
def _linear_matrix_offset(ell,ell_min):
    """Index of initial element in linear array of D matrices

    This gives the index based at the first element of the matrix with
    weight `ell`.  This is the quantity to be added to the result of
    `_linear_matrix_index`.

    This assumes that the input array corresponds to

    [[ell,mp,m] for ell in range(ell_min,ell_max+1)
                for mp in range(-ell,ell+1)
                for m in range(-ell,ell+1)]

    This result can be calculated from sympy as

    from sympy import symbols, summation
    ell_min,ell = symbols('ell_min,ell', integer=True)
    summation((2*ell + 1)**2, (ell, ell_min, ell-1))

    """
    return ( (4*ell**2-1)*ell - (4*ell_min**2-1)*ell_min ) // 3

@njit('int64(int64,int64)')
def _total_size_D_matrices(ell_min, ell_max):
    return ( ((4*ell_max+12)*ell_max+11)*ell_max + 3 - (4*ell_min**2-1)*ell_min ) // 3

@njit('complex128(complex128)')
def conjugate(z):
    return z.conjugate()

def Wigner_D_matrices(R, ell_min, ell_max):
    """Return linear array of Wigner D matrix elements for range of ell values

    Parameters
    ----------
    R : quaternion
        The rotor for the D matrices
    ell_min : int
        Lowest ell value included in array
    ell_max : int
        Highest ell value included in array

    Returns
    -------
    numpy.ndarray
        Linear array of all matrix elements

        The array is in the standard order, essentially of the form

            [D(ell,mp,m) for ell in range(ell_min, ell_max+1)
                         for mp in range(-ell,ell+1)
                         for m in range(-ell,ell+1)]

    See Also
    --------
    LMpM_total_size: Calculate total size of this array
    LMpM_index: Find index in this array of element (ell,mp,m)
    LMpM_range: Construct list of corresponding (ell,mp,m) values

    """
    matrices = np.empty((LMpM_total_size(ell_min,ell_max),), dtype=complex)
    _Wigner_D_matrices(R.a, R.b, ell_min, ell_max, matrices)
    return matrices

@njit('void(complex128, complex128, int64, int64, complex128[:])',
      locals={'Result1': complex128, 'Result2': complex128, 'Sum': complex128})
def _Wigner_D_matrices(Ra, Rb, ell_min, ell_max, matrices):
    """Main work function for computing Wigner D matrix elements

    This is the core function that does all the work in the
    computation, but it is strict about its input, and does not check
    them for validity.

    Input arguments
    ===============
    _Wigner_D_matrices(Ra, Rb, ell_min, ell_max, elements)

      * Ra, Rb are the complex components of the rotor
      * ell_min, ell_max are the limits of the matrices
      * matrix is a one-dimensional array of complex numbers to be
        filled with the elements of the matrices; the correct shape is
        assumed

    The `matrices` variable is needed because numba cannot create
    arrays at the moment, but this is modified in place, so after
    calling this function, the input array will contain the correct
    values.

    """

    # These constants are the recurring quantities in the computation
    # of the matrix elements, so we calculate them here just once
    ra,phia = cmath.polar(Ra)
    rb,phib = cmath.polar(Rb)

    if(ra<=epsilon):
        for ell in xrange(ell_min, ell_max+1):
            i_ell = _linear_matrix_offset(ell,ell_min)
            for i in xrange((2*ell+1)**2):
                matrices[i_ell+i] = 0j
            for mpmm in xrange(-ell,ell+1):
                i_mpmm = _linear_matrix_index(ell,mpmm,-mpmm)
                if (ell+mpmm)%2==0:
                    matrices[i_ell+i_mpmm] = Rb**(-2*mpmm)
                else:
                    matrices[i_ell+i_mpmm] = -Rb**(-2*mpmm)

    if(rb<=epsilon):
        for ell in xrange(ell_min, ell_max+1):
            i_ell = _linear_matrix_offset(ell,ell_min)
            for i in xrange((2*ell+1)**2):
                matrices[i_ell+i] = 0j
            for mpm in xrange(-ell,ell+1):
                i_mpm = _linear_matrix_diagonal_index(ell,mpm)
                matrices[i_ell+i_mpm] = Ra**(2*mpm)

    else:
        r__2 = complex(ra,-rb)**2
        phi = cmath.phase(r__2)
        for ell in xrange(ell_min, ell_max+1):
            i_ell = _linear_matrix_offset(ell,ell_min)
            for mp in xrange(-ell,1):
                for m in xrange(mp,-mp+1):
                    i_mpm = _linear_matrix_index(ell,mp,m)
                    Sum = 0.0+0.0j
                    for mpp in xrange(ell,-ell-1,-1):
                        # Note that the second Delta takes pi/2 as its argument, so
                        # we just take the conjugate transpose.  And since Delta is
                        # alsways real, it's just the transpose.  This is also good
                        # because it minimizes the jumping around when indexing the
                        # array.
                        Sum = (Delta(ell,mp,mpp)*Delta(ell,m,mpp)) + r__2*Sum
                    Result1 = cmath.exp(1j*(phia*(mp+m) + phib*(m-mp) - phi*(ell) + (m-mp)*np.pi/2)) * Sum
                    matrices[i_ell+i_mpm] = Result1
                    if(abs(m)!=abs(mp)):
                        Result2 = cmath.exp(1j*(-phia*(mp+m) + phib*(m-mp) - phi*(ell) - (m-mp)*np.pi/2)) * Sum
                        if (m+mp)%2==0:
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell+_linear_matrix_index(ell,-mp,-m)] = Result1.conjugate()
                            # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                            matrices[i_ell+_linear_matrix_index(ell,-m,-mp)] = Result2
                        else:
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell+_linear_matrix_index(ell,-mp,-m)] = -Result1.conjugate()
                            # D_{-m,-mp}(R) = (-1)^{mp+m} D_{mp,m}(\bar{R})
                            matrices[i_ell+_linear_matrix_index(ell,-m,-mp)] = -Result2
                        # D_{m,mp}(R) = \bar{D}_{mp,m}(\bar{R})
                        matrices[i_ell+_linear_matrix_index(ell,m,mp)] = Result2.conjugate()
                    elif(m!=0):
                        if (m+mp)%2==0:
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell+_linear_matrix_index(ell,-mp,-m)] = Result1.conjugate()
                        else:
                            # D_{-mp,-m}(R) = (-1)^{mp+m} \bar{D}_{mp,m}(R)
                            matrices[i_ell+_linear_matrix_index(ell,-mp,-m)] = -Result1.conjugate()
