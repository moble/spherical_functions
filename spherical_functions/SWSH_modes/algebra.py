# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Modes class.  In
### particular, they assume that the first argument, `self` is an instance of Modes.  They should
### probably not be used outside of that class.

import numpy as np
from .. import LM_total_size, LM_index
from ..multiplication import _multiplication_helper


def conj(self, inplace=False):
    return self.conjugate(inplace=inplace)


def conjugate(self, inplace=False):
    """Return Modes object corresponding to conjugated function

    The operations of conjugation and decomposition into mode weights do not commute.  That is,
    the modes of a conjugated function do not equal the conjugated modes of a function.  So,
    rather than simply returning the conjugate of the data from this Modes object, this function
    returns a Modes object containing the data for the conjugated function.

    If `inplace` is True, then the operation is performed in place, modifying this Modes object
    itself.  Note that some copying is still needed, but only for 2 modes at a time, and those
    copies are freed after this function returns.

    Here is the derivation:

        f = sum(f{s, l,m} * {s}Y{l,m} for l, m in LM)
        conjugate(f) = sum(conjugate(f{s, l,m}) * conjugate({s}Y{l,m}))
                     = sum(conjugate(f{s, l,m}) * (-1)**(s+m) * {-s}Y{l,-m})
                     = sum((-1)**(s+m) * conjugate(f{s, l, -m}) * {-s}Y{l,m})

        conjugate(f){s', l',m'} = integral(
            sum((-1)**(s+m) * conjugate(f{s,l,-m}) * {-s}Y{l,m}) * {s'}Y{l',m'},
            dR  # integration over rotation group
        )
        = sum((-1)**(s+m) * conjugate(f{s,l,-m}) * delta_{-s, s'} * delta_{l, l'} * delta_{m, m'})
        = (-1)**(s'+m') * conjugate(f{-s', l', -m'})

    The result is this:

        conjugate(f){s, l, m} = (-1)**(s+m) * conjugate(f{-s, l, -m})

    """
    s = self.view(np.ndarray)
    c = s if inplace else np.zeros_like(s)
    for ell in range(abs(self.s), self.ell_max+1):
        i = LM_index(ell, 0, self.ell_min)
        if self.s%2 == 0:
            c[..., i] = np.conjugate(s[..., i])
        else:
            c[..., i] = -np.conjugate(s[..., i])
        for m in range(1, ell+1):
            i_p, i_n = LM_index(ell, m, self.ell_min), LM_index(ell, -m, self.ell_min)
            if (self.s+m)%2 == 0:
                c[..., i_p], c[..., i_n] = np.conjugate(s[..., i_n]), np.conjugate(s[..., i_p])
            else:
                c[..., i_p], c[..., i_n] = -np.conjugate(s[..., i_n]), -np.conjugate(s[..., i_p])
    if inplace:
        self._metadata['spin_weight'] = -self.s
        return self
    return type(self)(c, s=-self.s, ell_min=self.ell_min, ell_max=self.ell_max)

conj.__doc__ = conjugate.__doc__


@property
def bar(self):
    return self.conjugate()
bar.__doc__ = conjugate.__doc__


def real(self, inplace=False):
    """Return Modes object corresponding to real-valued function

    Note that this only makes sense for functions of spin weight zero; other spins will raise
    ValueErrors.

    The condition that a function `f` be real is given by

        f{l, m} = conjugate(f){l, m} = (-1)**(m) * conjugate(f{l, -m})

    [Note that conj(f){l, m} != conj(f{l, m}).  See Modes.conjugate docstring.]

    We enforce that condition by essentially averaging the two modes:

        f{l, m} = (f{l, m} + (-1)**m * conjugate(f{l, -m})) / 2

    """
    if self.s != 0:
        raise ValueError("The real part of a function with non-zero spin weight is meaningless")
    s = self.view(np.ndarray)
    c = s if inplace else np.zeros_like(s)
    for ell in range(abs(self.s), self.ell_max+1):
        i = LM_index(ell, 0, self.ell_min)
        c[..., i] = np.real(s[..., i])
        for m in range(1, ell+1):
            i_p, i_n = LM_index(ell, m, self.ell_min), LM_index(ell, -m, self.ell_min)
            if m%2 == 0:
                c[..., i_p] = (s[..., i_p] + np.conjugate(s[..., i_n])) / 2
                c[..., i_n] = np.conjugate(c[..., i_p])
            else:
                c[..., i_p] = (s[..., i_p] - np.conjugate(s[..., i_n])) / 2
                c[..., i_n] = -np.conjugate(c[..., i_p])
    if inplace:
        return self
    return type(self)(c, s=self.s, ell_min=self.ell_min, ell_max=self.ell_max)


def imag(self, inplace=False):
    """Return Modes object corresponding to imaginary-valued function

    Note that this only makes sense for functions of spin weight zero; other spins will raise
    ValueErrors.

    The condition that a function `f` be imaginary is given by

        f{l, m} = -conjugate(f){l, m} = (-1)**(m+1) * conjugate(f{l, -m})

    [Note that conj(f){l, m} != conj(f{l, m}).  See Modes.conjugate docstring.]

    We enforce that condition by essentially averaging the two modes:

        f{l, m} = (f{l, m} - (-1)**m * conjugate(f{l, -m})) / 2

    """
    if self.s != 0:
        raise ValueError("The imaginary part of a function with non-zero spin weight is meaningless")
    s = self.view(np.ndarray)
    c = s if inplace else np.zeros_like(s)
    for ell in range(abs(self.s), self.ell_max+1):
        i = LM_index(ell, 0, self.ell_min)
        c[..., i] = 1j * np.imag(s[..., i])
        for m in range(1, ell+1):
            i_p, i_n = LM_index(ell, m, self.ell_min), LM_index(ell, -m, self.ell_min)
            if m%2 == 0:
                c[..., i_p] = (s[..., i_p] - np.conjugate(s[..., i_n])) / 2
                c[..., i_n] = -np.conjugate(c[..., i_p])
            else:
                c[..., i_p] = (s[..., i_p] + np.conjugate(s[..., i_n])) / 2
                c[..., i_n] = np.conjugate(c[..., i_p])
    if inplace:
        return self
    return type(self)(c, s=self.s, ell_min=self.ell_min, ell_max=self.ell_max)


def norm(self):
    return np.linalg.norm(self.view(np.ndarray), axis=-1)


def add(self, other, subtraction=False):
    if isinstance(other, type(self)):
        if self.s != other.s:
            raise ValueError(f"Cannot add modes with different spin weights ({self.s} and {other.s})")
        s = self.s
        ell_min = min(self.ell_min, other.ell_min)
        ell_max = max(self.ell_max, other.ell_max)
        shape = np.broadcast(self[..., 0], other[..., 0]).shape + (LM_total_size(ell_min, ell_max),)
        result = np.zeros(shape, dtype=np.complex_)
        i_s1 = LM_total_size(ell_min, self.ell_min-1)
        i_s2 = i_s1+LM_total_size(self.ell_min, self.ell_max)
        i_o1 = LM_total_size(ell_min, other.ell_min-1)
        i_o2 = i_o1+LM_total_size(other.ell_min, other.ell_max)
        result[..., i_s1:i_s2] = self.view(np.ndarray)
        if subtraction:
            result[..., i_o1:i_o2] -= other.view(np.ndarray)
        else:
            result[..., i_o1:i_o2] += other.view(np.ndarray)
        return type(self)(result, s=s, ell_min=ell_min, ell_max=ell_max)
    elif subtraction:
        return np.subtract(self, other)
    else:
        return np.add(self, other)


def subtract(self, other):
    if isinstance(other, type(self)):
        if self.s != other.s:
            raise ValueError(f"Cannot subtract modes with different spin weights ({self.s} and {other.s})")
        return self.add(other, True)
    else:
        return np.subtract(self, other)


def multiply(self, other, truncate=False):
    if isinstance(other, type(self)):
        if truncate is True:  # Note that we really do want this here; it's usually bad to test "is True"
            truncate = max(self.ell_max, other.ell_max)
        s = self.view(np.ndarray)
        o = other.view(np.ndarray)
        new_s = self.s + other.s
        new_ell_min = 0
        new_ell_max = truncate or self.ell_max + other.ell_max
        new_shape = np.broadcast(s[..., 0], o[..., 0]).shape + (LM_total_size(new_ell_min, new_ell_max),)
        new = np.zeros(new_shape, dtype=np.complex_)
        new = _multiplication_helper(s, self.ell_min, self.ell_max, self.s,
                                     o, other.ell_min, other.ell_max, other.s,
                                     new, new_ell_min, new_ell_max, new_s)
        return type(self)(new, s=new_s, ell_min=new_ell_min, ell_max=new_ell_max)
    else:
        return self * other


def divide(self, other):
    if isinstance(other, type(self)):
        raise ValueError(f"Cannot divide one Modes object by another")
    else:
        return self / other
