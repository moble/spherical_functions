# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Modes class.  In
### particular, they assume that the first argument, `self` is an instance of Modes.  They should
### probably not be used outside of that class.

import numpy as np
from .. import LM_total_size
from ..multiplication import _multiplication_helper


def __array_ufunc__(self, ufunc, method, *args, out=None, **kwargs):
    # These are required for basic support, but can be more-or-less passed through because they return bools
    if ufunc in [np.not_equal, np.equal, np.isfinite, np.isinf, np.isnan]:
        args = [arg.view(np.ndarray) if isinstance(arg, type(self)) else arg for arg in args]
        kwargs['out'] = out
        return ufunc(*args, **kwargs)

    # Here are all the functions we will support directly; all other ufuncs are probably meaningless
    elif ufunc not in [np.positive, np.negative, np.add, np.subtract,
                       np.multiply, np.divide, np.true_divide,
                       np.conj, np.conjugate, np.absolute]:
        return NotImplemented

    # We will not be supporting any more ufunc keywords
    if kwargs:
        raise NotImplementedError(f"Unrecognized arguments to {type(self)}.__array_ufunc__: {kwargs}")

    if ufunc in [np.positive, np.negative]:
        result = out or np.zeros(self.shape, dtype=np.complex_)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, type(self)):
            result = result.view(np.ndarray)
        ufunc(self.view(np.ndarray), out=result)
        if out is None:
            result = type(self)(result, s=self.s, ell_min=self.ell_min, ell_max=self.ell_max)
        elif isinstance(out[0], type(self)):
            out[0]._s = self.s
            # out[0]._ell_min = self.ell_min
            out[0]._ell_max = self.ell_max

    elif ufunc in [np.add, np.subtract]:
        if isinstance(args[0], type(self)) and isinstance(args[1], type(self)):
            m1, m2 = args[:2]
            if m1.s != m2.s:
                return NotImplemented
            s = m1.s
            ell_min = min(m1.ell_min, m2.ell_min)
            ell_max = max(m1.ell_max, m2.ell_max)
            shape = np.broadcast(m1[..., 0], m2[..., 0]).shape + (LM_total_size(ell_min, ell_max),)
            result = out or np.zeros(shape, dtype=np.complex_)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, type(self)):
                result = result.view(np.ndarray)
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
                result = type(self)(result, s=s, ell_min=ell_min, ell_max=ell_max)
            elif isinstance(out[0], type(self)):
                out[0]._s = s
                # out[0]._ell_min = ell_min
                out[0]._ell_max = ell_max
        elif isinstance(args[0], type(self)):
            modes = args[0]
            scalars = np.asanyarray(args[1])
            if modes.s != 0 or not modes._check_broadcasting(scalars):
                return NotImplemented
            result = ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out)
            if out is None:
                result = type(self)(result, self.s, self.ell_min, self.ell_max)
        elif isinstance(args[1], type(self)):
            scalars = np.asanyarray(args[0])
            modes = args[1]
            if modes.s != 0 or not modes._check_broadcasting(scalars, reverse=True):
                return NotImplemented
            result = ufunc(scalars[..., np.newaxis], modes.view(np.ndarray), out=out)
            if out is None:
                result = type(self)(result, self.s, self.ell_min, self.ell_max)
        else:
            return NotImplemented

    elif ufunc is np.multiply:
        if isinstance(args[0], type(self)) and isinstance(args[1], type(self)):
            s = args[0].view(np.ndarray)
            o = args[1].view(np.ndarray)
            result_s = args[0].s + args[1].s
            result_ell_min = 0
            result_ell_max = args[0].ell_max + args[1].ell_max
            result_shape = np.broadcast(s[..., 0], o[..., 0]).shape + (LM_total_size(result_ell_min, result_ell_max),)
            result = out or np.zeros(result_shape, dtype=np.complex_)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, type(self)):
                result = result.view(np.ndarray)
            _multiplication_helper(s, args[0].ell_min, args[0].ell_max, args[0].s,
                                   o, args[1].ell_min, args[1].ell_max, args[1].s,
                                   result, result_ell_min, result_ell_max, result_s)
            if out is None:
                result = type(self)(result, s=result_s, ell_min=result_ell_min, ell_max=result_ell_max)
            elif isinstance(out[0], type(self)):
                out[0]._s = result_s
                # out[0]._ell_min = result_ell_min
                out[0]._ell_max = result_ell_max
        elif isinstance(args[0], type(self)):
            modes = args[0]
            scalars = np.asanyarray(args[1])
            if not modes._check_broadcasting(scalars):
                return NotImplemented
            result = ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out)
            if out is None:
                result = type(self)(result, self.s, self.ell_min, self.ell_max)
        elif isinstance(args[1], type(self)):
            scalars = np.asanyarray(args[0])
            modes = args[1]
            if not modes._check_broadcasting(scalars, reverse=True):
                return NotImplemented
            result = ufunc(scalars[..., np.newaxis], modes.view(np.ndarray), out=out)
            if out is None:
                result = type(self)(result, self.s, self.ell_min, self.ell_max)
        else:
            return NotImplemented

    elif ufunc in [np.divide, np.true_divide]:
        if isinstance(args[1], type(self)) or not isinstance(args[0], type(self)):
            return NotImplemented
        modes = args[0]
        scalars = np.asanyarray(args[1])
        if not modes._check_broadcasting(scalars):
            return NotImplemented
        result = ufunc(modes.view(np.ndarray), scalars[..., np.newaxis], out=out)
        if out is None:
            result = type(self)(result, self.s, self.ell_min, self.ell_max)

    elif ufunc in [np.conj, np.conjugate]:
        raise NotImplementedError()

    elif ufunc is np.absolute:
        return args[0].norm()

    else:
        # I thought we filtered everything else out...
        raise NotImplementedError(f"{type(self)}.__array_ufunc__ has reached a point it should not have for ufunc {ufunc}")

    if result is NotImplemented:
        return NotImplemented

    if method == 'at':
        return

    return result
