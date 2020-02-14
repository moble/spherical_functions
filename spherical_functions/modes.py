import math
import numpy as np
import spinsfast
from . import LM_total_size, Wigner3j, LM_index, LM_deduce_ell_max
from .multiplication import _multiplication_helper


class Modes(np.ndarray):
    """Object to store SWHS modes

    This class subclasses numpy's ndarray object, so that it should act like a numpy array in many
    respects, even with functions like np.zeros_like.

    Note that the number of dimensions is arbitrary (as long as it is greater than 0), but the modes
    must be stored in the last axis.  For example, a SWSH function of time may be stored as a 2-d
    array where the first axis represents different times, and the second axis represents the mode
    weights at each instant of time.

    This class also does three important things that are unlike numpy arrays:

    1) It tracks the spin weight and minimum and maximum ell values stored in this data.

    2) It provides additional convenience methods, like `index` to find the index of a particular
       (ell, m) mode; a `grid` method to convert modes to the SWSH values on a grid over the sphere
       (while correctly handling additional dimensions); and the various derivative operators,
       including $\eth$ and $\bar{\eth}$.

    3) It overrides most of numpy's "universal functions" (ufuncs) to work appropriately for
       spin-weighted functions.  Specifically, these ufuncs are interpreted as acting on the
       spin-weighted function itself, rather than just the mode weights.  Most importantly, we have
       these

       a) Conjugating a Modes object will result in a new Modes object that represents the
          conjugated spin-weighted function, rather than simply conjugating the mode-weights of the
          function.

       b) Multiplying two Modes objects will result in a new Modes object that represents the
          pointwise product of the functions themselves (and will correctly have spin weight given
          by the sum of the spin weights of the first two functions), rather than the product of the
          mode weights.  Division is only permitted when the divisor is a constant.

       c) Addition (and subtraction) is permitted for functions of the same spin weight, but it does
          not make sense to add (or subtract) functions with different spin weights, so any attempt
          to do so raises a ValueError.  [Note that adding a constant is equivalent to adding a
          function of spin weight zero, and is treated in the same way.]

       d) The "absolute" ufunc does not return the absolute value of each mode weight (which is
          almost certainly meaningless); it returns the L2 norm of spin-weighted function over the
          sphere -- which happens to equal the sum of the squares of the absolute values of the mode
          weights.

       Numerous other ufuncs -- such as log, exp, trigonometric ufuncs, bit-twiddling ufuncs, and so
       on -- are disabled because they don't make sense when applied to functions.

    It is possible to treat the underlying data of a Modes object `modes` as an ordinary numpy array
    by taking `modes.view(np.ndarray)`.  However, it is hoped that this class already performs all
    reasonable operations.  If you find a missing feature that requires you to resort to this,
    please feel free to open an issue in this project's github page to discuss it.

    """


    def __new__(cls, input_array, s=None, ell_min=0, ell_max=None):
        input_array = np.asarray(input_array)
        if input_array.dtype != np.complex:
            raise ValueError(f"Input array must have dtype `complex`; dtype is `{input_array.dtype}`.\n            "
                             +"You can use `input_array.view(complex)` if the data are\n            "
                             +"stored as consecutive real and imaginary parts.")
        ell_max = ell_max or LM_deduce_ell_max(input_array.shape[-1], ell_min)
        if input_array.shape[-1] != LM_total_size(ell_min, ell_max):
            raise ValueError(f"Input array has shape {input_array.shape}.  Its last dimension should "
                             +f"have size {LM_total_size(ell_min, ell_max)},\n            "
                             +f"to be consistent with the input ell_min ({ell_min}) and ell_max ({ell_max})")
        if ell_min == 0:
            obj = input_array.view(cls)
        else:
            insertion_indices = [0,]*LM_total_size(0, ell_min-1)
            obj = np.insert(input_array, insertion_indices, 0.0, axis=-1).view(cls)
        obj[..., :LM_total_size(0, abs(s)-1)] = 0.0
        obj._s = s
        obj._ell_max = ell_max
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._s = getattr(obj, 's', None)
        self._ell_max = getattr(obj, 'ell_max', None)

    @property
    def s(self):
        """Spin weight of this Modes object"""
        return self._s
    
    @property
    def ell_min(self):
        """Smallest ell value stored in data [need not equal abs(s)]"""
        return 0
    
    @property
    def ell_max(self):
        """Largest ell value stored in data"""
        return self._ell_max

    def index(self, ell, m):
        """Return index of (ell, m) mode in data

        Note that the modes are always stored in the last axis of the data, so if your Modes object
        `modes` has (or even just might have) more than one dimension, you might extract a
        particular mode weight with something like this:

            i = modes.index(ell, m)
            ell_m_modes = modes[..., i]

        The ellipsis just represents all other dimensions (even if there are none).

        """
        if ell < abs(self.s) or ell < abs(m):
            raise ValueError(f"Bad index (ell, m)=({ell}, {m}) for spin weight s={self.s}")
        if ell < self.ell_min or ell > self.ell_max:
            raise ValueError(f"Requested ell index {ell} outside bounds of this data ({self.ell_min, self.ell_max})")
        return LM_index(ell, m, self.ell_min)
    
    def grid(self, n_theta=None, n_phi=None):
        n_theta = n_theta or 2*self.ell_max+1
        n_phi = n_phi or n_theta
        return spinsfast.salm2map(self.view(np.ndarray), self.s, self.ell_max, n_theta, n_phi)

    def add(self, other, subtraction=False):
        if isinstance(other, Modes):
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
            return Modes(result, s=s, ell_min=ell_min, ell_max=ell_max)
        elif subtraction:
            return np.subtract(self, other)
        else:
            return np.add(self, other)
    
    def subtract(self, other):
        if isinstance(other, Modes):
            if self.s != other.s:
                raise ValueError(f"Cannot subtract modes with different spin weights ({self.s} and {other.s})")
            return self.add(other, True)
        else:
            return np.subtract(self, other)
    
    def multiply(self, other, truncate=False):
        if isinstance(other, Modes):
            s = self.view(np.ndarray)
            o = other.view(np.ndarray)
            new_s = self.s + other.s
            new_ell_min = 0
            new_ell_max = max(self.ell_max, other.ell_max) if truncate else self.ell_max + other.ell_max
            new_shape = np.broadcast(s[..., 0], o[..., 0]).shape + (LM_total_size(new_ell_min, new_ell_max),)
            new = np.zeros(new_shape, dtype=np.complex_)
            new = _multiplication_helper(s, self.ell_min, self.ell_max, self.s,
                                         o, other.ell_min, other.ell_max, other.s,
                                         new, new_ell_min, new_ell_max, new_s)
            return Modes(new, s=new_s, ell_min=new_ell_min, ell_max=new_ell_max)
        else:
            return self * other
    
    def divide(self, other):
        if isinstance(other, Modes):
            raise ValueError(f"Cannot divide one Modes object by another")
        else:
            return self / other
    
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
            self._s *= -1
            return self
        return Modes(c, s=-self.s, ell_min=self.ell_min, ell_max=self.ell_max)

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
        return Modes(c, s=self.s, ell_min=self.ell_min, ell_max=self.ell_max)

    def Lz(self):
        """Left Lie derivative with respect to rotation about z

        The left Lie derivative of a function f(Q) over the unit
        quaternions with respect to a generator of rotation g is
        defined as

            Lg(f){Q} = -0.5j df{exp(t*g) * Q} / dt |t=0

        This agrees with the usual angular-momentum operators familiar
        from spherical-harmonic theory, and reduces to it when the
        function has spin weight 0, but also applies to functions of
        general spin weight.  In terms of the SWSHs, we can write the
        action of Lz as

            Lz {s}Y{l,m} = m * {s}Y{l,m}

        """
        d = self.copy()
        s = self.view(np.ndarray)
        for ell in range(abs(self.s), self.ell_max+1):
            for m in range(-ell, ell+1):
                d[..., d.index(ell, m)] = m * s[..., self.index(ell, m)]
        return d
    
    def Lplus(self):
        """Raising operator for Lz

        We define Lplus to be the raising operator for the left Lie
        derivative with respect to rotation about z, Lz.  By
        definition, this means that [Lz, Lplus] = Lplus, which allows
        us to derive Lplus = Lx + 1j * Ly.  In terms of the SWSHs, we
        can write the action of Lplus as

            Lplus {s}Y{l,m} = sqrt((l-m)*(l+m+1)) {s}Y{l,m+1}

        Consequently, the modes of a function are affected as

            {Lplus f}{s, l, m} = sqrt((l+m)*(l-m+1)) * f{s,l,m-1}
        
        """
        # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
        # Lplus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Lplus D{l,m,-s}
        #                 = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l-m)(l+m+1)) D{l,m+1,-s}
        #                 = sqrt((l-m)(l+m+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m+1,-s}
        #                 = sqrt((l-m)(l+m+1)) {s}Y{l,m+1}
        # {L+ f}{s', l', m'}
        #    = integral(L+ f {s'}Ybar{l',m'})  # Integral over rotation group
        #    = integral(L+ sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
        #    = sum(f{s,l,m} integral(L+ {s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(f{s,l,m} integral(sqrt((l-m)(l+m+1)) {s}Y{l,m+1} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l-m)(l+m+1)) f{s,l,m} integral({s}Y{l,m+1} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l-m)(l+m+1)) f{s,l,m} delta{s, s'} delta{m+1, m'} delta{l, l'}
        #    = sqrt((l'-(m'-1))(l'+(m'-1)+1)) f{s',l',m'-1}
        #    = sqrt((l'+m')(l'-m'+1)) f{s',l',m'-1}
        # {L+ f}{s, l, m} = sqrt((l+m)(l-m+1)) f{s,l,m-1}
        d = np.zeros_like(self)
        s = self.view(np.ndarray)
        for ell in range(abs(self.s), self.ell_max+1):
            d[..., self.index(ell, -ell)] = 0.0
            for m in range(-ell+1, ell+1):
                d[..., d.index(ell, m)] = math.sqrt((ell+m)*(ell-m+1)) * s[..., self.index(ell, m-1)]
        return d
    
    def Lminus(self):
        """Lowering operator for Lz

        We define Lminus to be the lowering operator for the left Lie
        derivative with respect to rotation about z, Lz.  By
        definition, this means that [Lz, Lminus] = -Lminus, which allows
        us to derive Lminus = Lx - 1j * Ly.  In terms of the SWSHs, we
        can write the action of Lminus as

            Lminus {s}Y{l,m} = sqrt((l+m)*(l-m+1)) * {s}Y{l,m-1}

        Consequently, the modes of a function are affected as

            {Lminus f}{s, l, m} = sqrt((l-m)*(l+m+1)) * f{s,l,m+1}

        """
        # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
        # Lminus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Lminus D{l,m,-s}
        #                  = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l+m)(l-m+1)) D{l,m-1,-s}
        #                  = sqrt((l+m)(l-m+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m-1,-s}
        #                  = sqrt((l+m)(l-m+1)) {s}Y{l,m-1}
        # {L- f}{s', l', m'}
        #    = integral(L- f {s'}Ybar{l',m'})  # Integral over rotation group
        #    = integral(L- sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
        #    = sum(f{s,l,m} integral(L- {s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(f{s,l,m} integral(sqrt((l+m)(l-m+1)) {s}Y{l,m-1} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l+m)(l-m+1)) f{s,l,m} integral({s}Y{l,m-1} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l+m)(l-m+1)) f{s,l,m} delta{s, s'} delta{m-1, m'} delta{l, l'}
        #    = sqrt((l'+(m'+1))(l'-(m'+1)+1)) f{s',l',m'+1}
        #    = sqrt((l'-m')(l'+m'+1)) f{s',l',m'+1}
        # {L- f}{s, l, m} = sqrt((l-m)(l+m+1)) f{s,l,m+1}
        d = np.zeros_like(self)
        s = self.view(np.ndarray)
        for ell in range(abs(self.s), self.ell_max+1):
            for m in range(-ell, ell):
                d[..., self.index(ell, m)] = math.sqrt((ell-m)*(ell+m+1)) * s[..., self.index(ell, m+1)]
            d[..., self.index(ell, ell)] = 0.0
        return d
    
    def Rz(self):
        """Right Lie derivative with respect to rotation about z

        The right Lie derivative of a function f(Q) over the unit
        quaternions with respect to a generator of rotation g is
        defined as

            Rg(f){Q} = -0.5j df{Q * exp(t*g)} / dt |t=0

        This is unlike the usual angular-momentum operators Lz, etc.,
        familiar from spherical-harmonic theory because the
        exponential is on the right-hand side of the argument.  This
        operator is less common in physics because it represents the
        dependence of the function on the choice of frame.  In terms
        of the SWSHs, we can write the action of Rz as

            Rz {s}Y{l,m} = -s * {s}Y{l,m}

        Equivalently, the modes of a function are affected as

            {Rz f} {s,l,m} = -s * f{s,l,m}

        Note the unfortunate sign of `s`, which seems to be opposite
        to what we expect, and arises from the choice of definition of
        `s` in the original paper by Newman and Penrose.

        """
        # {Rzf}{s', l', m'}
        #    = integral(Rz f {s'}Ybar{l',m'})  # Integral over rotation group
        #    = integral(Rz sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
        #    = sum(f{s,l,m} integral(Rz {s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(f{s,l,m} integral(-s {s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(-s f{s,l,m} integral({s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(-s f{s,l,m} delta{s, s'} delta{m, m'} delta{l, l'}
        #    = -s f{s',l',m'}
        # {Rzf}{s, l, m} = -s f{s,l,m}
        return Modes(-self.s * self.view(np.ndarray), s=self.s, ell_min=self.ell_min, ell_max=self.ell_max)
    
    def Rplus(self):
        """Raising operator for Rz

        We define Rplus to be the raising operator for the right Lie
        derivative with respect to rotation about z, Rz.  By
        definition, this means that [Rz, Rplus] = Rplus, which allows
        us to derive Rplus = Rx - 1j * Ry.  In terms of the SWSHs, we
        can write the action of Rplus as

            Rplus {s}Y{l,m} = sqrt((l+s)(l-s+1)) {s-1}Y{l,m}

        Consequently, the modes of a function are affected as

            {Rplus f} {s,l,m} = sqrt((l-s)(l+s+1)) f{s+1,l,m}

        Again, because of the unfortunate choice of the sign of `s`
        made in the original paper by Newman and Penrose, this looks
        like a lowering operator for `s`.  But it really is a raising
        operator for Rz, and raises the eigenvalue of the
        corresponding Wigner matrix - though that lowers the value of
        `s`.

        """
        # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
        # Rplus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Rplus D{l,m,-s}
        #                 = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l+s)(l-s+1)) D{l,m,-s+1}
        #                 = sqrt((l+s)(l-s+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s+1}
        #                 = sqrt((l+s)(l-s+1)) {s-1}Y{l,m}
        # {R+f}{s', l', m'}
        #    = integral(R+ f {s'}Ybar{l',m'})  # Integral over rotation group
        #    = integral(R+ sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
        #    = sum(f{s,l,m} integral(R+ {s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(f{s,l,m} integral(sqrt((l+s)(l-s+1)) {s-1}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l+s)(l-s+1)) f{s,l,m} integral({s-1}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l+s)(l-s+1)) f{s,l,m} delta{s-1, s'} delta{m, m'} delta{l, l'}
        #    = sqrt((l'+s'+1)(l'-(s'+1)+1) f{s'+1,l',m'}
        # {R+f}{s, l, m} = sqrt((l-s)(l+s+1)) f{s+1,l,m}
        d = Modes(np.zeros_like(self.view(np.ndarray)), s=self.s-1, ell_min=min(abs(self.s-1), self.ell_min), ell_max=self.ell_max)
        s = self.view(np.ndarray)
        for ell in range(max(abs(d.s), abs(self.s)), d.ell_max+1):
            if ell >= self.ell_min:
                d[..., d.index(ell, -ell):d.index(ell, ell)+1] = (
                    math.sqrt((ell-d.s)*(ell+d.s+1))
                    * s[..., self.index(ell, -ell):self.index(ell, ell)+1]
                )
        return d
    
    def Rminus(self):
        """Lowering operator for Rz

        We define Rminus to be the lowering operator for the right Lie
        derivative with respect to rotation about z, Rz.  By
        definition, this means that [Rz, Rminus] = -Rminus, which
        allows us to derive Rminus = Rx + 1j * Ry.  In terms of the
        SWSHs, we can write the action of Rminus as

            Rminus {s}Y{l,m} = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}

        Consequently, the modes of a function are affected as

            {Rminus f} {s,l,m} = sqrt((l+s)(l-s+1)) f{s-1,l,m}

        Again, because of the unfortunate choice of the sign of `s`
        made in the original paper by Newman and Penrose, this looks
        like a raising operator for `s`.  But it really is a lowering
        operator for Rz, and lowers the eigenvalue of the
        corresponding Wigner matrix - though that raises the value of
        `s`.

        """
        # sYlm = (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s}
        # Rminus {s}Y{l,m} = (-1)**s sqrt((2ell+1)/(4pi)) Rminus D{l,m,-s}
        #                  = (-1)**s sqrt((2ell+1)/(4pi)) sqrt((l-s)(l+s+1)) D{l,m,-s-1}
        #                  = sqrt((l-s)(l+s+1)) (-1)**s sqrt((2ell+1)/(4pi)) D{l,m,-s-1}
        #                  = sqrt((l-s)(l+s+1)) {s+1}Y{l,m}
        # {R- f}{s', l', m'}
        #    = integral(R- f {s'}Ybar{l',m'})  # Integral over rotation group
        #    = integral(R- sum(f{s,l,m}{s}Y{l,m}) {s'}Ybar{l',m'})
        #    = sum(f{s,l,m} integral(R- {s}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(f{s,l,m} integral(sqrt((l-s)(l+s+1)) {s+1}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l-s)(l+s+1)) f{s,l,m} integral({s+1}Y{l,m} {s'}Ybar{l',m'}))
        #    = sum(sqrt((l-s)(l+s+1)) f{s,l,m} delta{s+1, s'} delta{m, m'} delta{l, l'}
        #    = sqrt((l'-(s'-1))(l'+(s'-1)+1)) f{s'-1,l',m'}
        #    = sqrt((l'-s'+1)(l'+s')) f{s'-1,l',m'}
        # {R- f}{s, l, m} = sqrt((l+s)(l-s+1)) f{s-1,l,m}
        d = Modes(np.zeros_like(self.view(np.ndarray)), s=self.s+1, ell_min=min(abs(self.s+1), self.ell_min), ell_max=self.ell_max)
        s = self.view(np.ndarray)
        for ell in range(max(abs(d.s), abs(self.s)), d.ell_max+1):
            if ell >= self.ell_min:
                d[..., d.index(ell, -ell):d.index(ell, ell)+1] = (
                    math.sqrt((ell+d.s)*(ell-d.s+1))
                    * s[..., self.index(ell, -ell):self.index(ell, ell)+1]
                )
        return d

    def eth(self):
        return self.Rminus()

    def ethbar(self):
        return -self.Rplus()

    def norm(self):
        return np.linalg.norm(self.view(np.ndarray), axis=-1)

    def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
        # These are required for basic support, but can be more-or-less passed through because they return bools
        if ufunc in [np.not_equal, np.equal, np.isfinite, np.isinf, np.isnan]:
            args = [arg.view(np.ndarray) if isinstance(arg, Modes) else arg for arg in args]
            kwargs['out'] = out
            return ufunc(*args, **kwargs)

        # Here are all the functions we will support directly; all other ufuncs are probably meaningless
        elif ufunc not in [np.positive, np.negative, np.add, np.subtract,
                           np.multiply, np.divide, np.true_divide,
                           np.conj, np.conjugate, np.absolute]:
            return NotImplemented

        # We will not be supporting any more ufunc keywords
        if kwargs:
            raise NotImplementedError(f"Unrecognized arguments to Modes.__array_ufunc__: {kwargs}")

        def check_broadcasting(modes, scalar, reverse=False):
            try:
                if reverse:
                    np.broadcast(scalar, modes[..., 0])
                else:
                    np.broadcast(modes[..., 0], scalar)
            except ValueError:
                return False
            return True

        if ufunc in [np.positive, np.negative]:
            result = out or np.zeros(self.shape, dtype=np.complex_)
            ufunc(self.view(np.ndarray), out=result)
            if out is None:
                result = Modes(result, s=self.s, ell_min=self.ell_min, ell_max=self.ell_max)
            elif isinstance(out, Modes):
                out._s = self.s
                # out._ell_min = self.ell_min
                out._ell_max = self.ell_max
        elif ufunc in [np.add, np.subtract]:
            if isinstance(args[0], Modes) and isinstance(args[1], Modes):
                m1, m2 = args[:2]
                if m1.s != m2.s:
                    return NotImplemented
                s = m1.s
                ell_min = min(m1.ell_min, m2.ell_min)
                ell_max = max(m1.ell_max, m2.ell_max)
                shape = np.broadcast(m1[..., 0], m2[..., 0]).shape + (LM_total_size(ell_min, ell_max),)
                result = out or np.zeros(shape, dtype=np.complex_)
                i_s1 = LM_total_size(ell_min, m1.ell_min-1)
                i_s2 = i_s1+LM_total_size(m1.ell_min, m1.ell_max)
                i_o1 = LM_total_size(ell_min, m2.ell_min-1)
                i_o2 = i_o1+LM_total_size(m2.ell_min, m2.ell_max)
                result[..., i_s1:i_s2] = m1.view(np.ndarray)
                if ufunc is np.subtract:
                    result[..., i_o1:i_o2] -= m2.view(np.ndarray)
                else:
                    result[..., i_o1:i_o2] += m2.view(np.ndarray)
                if out is None:
                    result = Modes(result, s=s, ell_min=ell_min, ell_max=ell_max)
                elif isinstance(out, Modes):
                    out._s = s
                    # out._ell_min = ell_min
                    out._ell_max = ell_max
            elif isinstance(args[0], Modes):
                modes = args[0]
                scalars = np.asanyarray(args[1])
                if modes.s != 0 or not check_broadcasting(modes, scalars):
                    return NotImplemented
                result = ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out)
                if out is None:
                    result = Modes(result, self.s, self.ell_min, self.ell_max)
            elif isinstance(args[1], Modes):
                scalars = np.asanyarray(args[0])
                modes = args[1]
                if modes.s != 0 or not check_broadcasting(modes, scalars, reverse=True):
                    return NotImplemented
                result = ufunc(scalars[..., np.newaxis], modes.view(np.ndarray), out=out)
                if out is None:
                    result = Modes(result, self.s, self.ell_min, self.ell_max)
            else:
                return NotImplemented

        elif ufunc is np.multiply:
            if isinstance(args[0], Modes) and isinstance(args[1], Modes):
                s = args[0].view(np.ndarray)
                o = args[1].view(np.ndarray)
                result_s = args[0].s + args[1].s
                result_ell_min = 0
                result_ell_max = args[0].ell_max + args[1].ell_max
                result_shape = np.broadcast(s[..., 0], o[..., 0]).shape + (LM_total_size(result_ell_min, result_ell_max),)
                result = out or np.zeros(result_shape, dtype=np.complex_)
                _multiplication_helper(s, args[0].ell_min, args[0].ell_max, args[0].s,
                                       o, args[1].ell_min, args[1].ell_max, args[1].s,
                                       result, result_ell_min, result_ell_max, result_s)
                if out is None:
                    result = Modes(result, s=result_s, ell_min=result_ell_min, ell_max=result_ell_max)
                elif isinstance(out, Modes):
                    out._s = result_s
                    # out._ell_min = result_ell_min
                    out._ell_max = result_ell_max
            elif isinstance(args[0], Modes):
                modes = args[0]
                scalars = np.asanyarray(args[1])
                if not check_broadcasting(modes, scalars):
                    return NotImplemented
                result = ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out)
                if out is None:
                    result = Modes(result, self.s, self.ell_min, self.ell_max)
            elif isinstance(args[1], Modes):
                scalars = np.asanyarray(args[0])
                modes = args[1]
                if not check_broadcasting(modes, scalars, reverse=True):
                    return NotImplemented
                result = ufunc(scalars[..., np.newaxis], modes.view(np.ndarray), out=out)
                if out is None:
                    result = Modes(result, self.s, self.ell_min, self.ell_max)
            else:
                return NotImplemented

        elif ufunc in [np.divide, np.true_divide]:
            if isinstance(args[1], Modes) or not isinstance(args[0], Modes):
                return NotImplemented
            modes = args[0]
            scalars = np.asanyarray(args[1])
            if not check_broadcasting(modes, scalars):
                return NotImplemented
            result = ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out)
            if out is None:
                result = Modes(result, self.s, self.ell_min, self.ell_max)

        elif ufunc in [np.conj, np.conjugate]:
            raise NotImplementedError()

        elif ufunc is np.absolute:
            return args[0].norm()

        else:
            # I thought we filtered everything else out...
            raise NotImplementedError(f"Modes.__array_ufunc__ has reached a point it should not have for ufunc {ufunc}")

        if result is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        return result
    

Modes.conj.__doc__ = Modes.conjugate.__doc__
