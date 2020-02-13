import numpy as np
import spinsfast
from . import LM_total_size, Wigner3j, LM_index, LM_deduce_ell_max
from .multiplication import _multiplication_helper


class Modes(np.ndarray):

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
                             f"to be consistent with the input ell_min ({ell_min}) and ell_max ({ell_max})")
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
        return self._s
    
    @property
    def ell_min(self):
        return 0
    
    @property
    def ell_max(self):
        return self._ell_max

    def index(self, ell, m):
        return sf.LM_index(ell, m, self.ell_min)
    
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
        rather than simply returning the conjugate of the data from this Modes object, this
        function returns a Modes object containing the data for the conjugated function.

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
    
    def eth(self):
        raise NotImplementedError()
    
    def ethbar(self):
        raise NotImplementedError()
        
    def norm(self):
        return np.linalg.norm(self.view(np.ndarray), axis=-1)

    def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
        if ufunc in [np.add, np.subtract, np.multiply, np.divide, np.true_divide, np.floor_divide]:
            if isinstance(args[0], Modes) and isinstance(args[1], Modes):
                raise NotImplementedError()
            else:
                raise NotImplementedError()
        if ufunc in [np.conj, np.conjugate, np.absolute]:
            raise NotImplementedError()
        
        args_view = [
            arg_i.view(np.ndarray) if isinstance(arg_i, Modes)
            else arg_i
            for arg_i in args
        ]
        if out is not None:
            out_view = [
                out_i.view(np.ndarray) if isinstance(out_i, Modes)
                else out_i
                for out_i in out
            ]
            kwargs['out'] = out_view
        else:
            out_view = (None,) * ufunc.nout
            kwargs['out'] = out

        results = super(Modes, self).__array_ufunc__(ufunc, method, *args_view, **kwargs)

        if results is NotImplemented:
            return results
        
        if method == 'at':
            return

        if ufunc.nout == 1:
            results = (results,)
        
        results = tuple(np.asarray(result).view(Modes) if output is None else output
                        for result, output in zip(results, out_view))

        return results[0] if len(results) == 1 else results
    

Modes.conj.__doc__ = Modes.conjugate.__doc__
