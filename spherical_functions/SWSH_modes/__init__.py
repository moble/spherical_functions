# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import copy
import math
import numpy as np
import spinsfast
from .. import LM_total_size, Wigner3j, LM_index, LM_deduce_ell_max
from ..multiplication import _multiplication_helper


class Modes(np.ndarray):
    """Object to store SWHS modes

    This class subclasses numpy's ndarray object, so that it should act like a numpy array in many
    respects, even with functions like np.zeros_like.

    NOTE: The functions `np.copy(modes)` and `np.array(modes, copy=True)` return `ndarray` objects;
    they lose information about the SWSH attributes, and have different ufuncs.  If you wish to keep
    this information, use `modes.copy()`.  Also note that pickling works as expected, as do
    copy.copy and copy.deepcopy.

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

    Also, be aware that ndarrays also have various built-in methods that cannot be overridden very easily, such as max

    """

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __new__(cls, input_array, s, ell_min=0, ell_max=None):
        input_array = np.asarray(input_array).view(complex)
        if isinstance(s, dict):
            metadata = copy.deepcopy(s)
            if 'spin_weight' not in metadata:
                raise ValueError("When second argument is a dict, it must contain the key 'spin_weight'")
            if ell_max is not None and 'ell_max' in metadata and ell_max != metadata['ell_max']:
                raise ValueError("The 'ell_max' value should only be specified once; it is specified in the second argument and as a keyword argument")
            s = metadata['spin_weight']
            ell_max = metadata.get('ell_max', ell_max)
        else:
            metadata = None
        ell_max = ell_max or LM_deduce_ell_max(input_array.shape[-1], ell_min)
        if input_array.shape[-1] != LM_total_size(ell_min, ell_max):
            raise ValueError(f"Input array has shape {input_array.shape} when viewed as a complex array.\n            "
                             +f"Its last dimension should have size {LM_total_size(ell_min, ell_max)}, "
                             +f"for consistency with the input ell_min ({ell_min}) and ell_max ({ell_max}).")
        if ell_min == 0:
            obj = input_array.view(cls)
        else:
            insertion_indices = [0,]*LM_total_size(0, ell_min-1)
            obj = np.insert(input_array, insertion_indices, 0.0, axis=-1).view(cls)
        obj[..., :LM_total_size(0, abs(s)-1)] = 0.0
        obj._metadata = metadata or {
            'spin_weight': s,
            'ell_max': ell_max,
        }
        return obj

    # https://numpy.org/doc/1.18/user/basics.subclassing.html
    def __array_finalize__(self, obj):
        if obj is None: return
        self._metadata = copy.deepcopy(getattr(obj, '_metadata', {'spin_weight': None, 'ell_max': None}))

    # For pickling
    def __reduce__(self):
        state = super(Modes, self).__reduce__()
        new_attributes = state[2] + (self._metadata,)
        return (state[0], state[1], new_attributes)

    # For unpickling
    def __setstate__(self, state):
        self._metadata = copy.deepcopy(state[-1])
        super(Modes, self).__setstate__(state[:-1])

    @property
    def s(self):
        """Spin weight of this Modes object"""
        return self._metadata['spin_weight']

    @property
    def ell_min(self):
        """Smallest ell value stored in data [need not equal abs(s)]"""
        return 0

    @property
    def ell_max(self):
        """Largest ell value stored in data"""
        return self._metadata['ell_max']

    @property
    def n_modes(self):
        """Number of elements along the last axis"""
        return self.shape[-1]

    from .algebra import (
        conj, conjugate, bar, real, norm,
        add, subtract, multiply, divide
    )

    from .derivatives import (
        Lsquared, Lz, Lplus, Lminus,
        Rsquared, Rz, Rplus, Rminus,
        eth, ethbar
    )

    from .utilities import (
        index, grid, _check_broadcasting
    )

    from .ufuncs import __array_ufunc__
