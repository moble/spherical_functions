# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the Modes class.  In
### particular, they assume that the first argument, `self` is an instance of Modes.  They should
### probably not be used outside of that class.


def index(self, ell, m):
    """Return index of (ell, m) mode in data

    Note that the modes are always stored in the last axis of the data, so if your Modes object
    `modes` has (or even just might have) more than one dimension, you might extract a
    particular mode weight with something like this:

        i = modes.index(ell, m)
        ell_m_modes = modes[..., i]

    The ellipsis just represents all other dimensions (even if there are none).

    """
    from .. import LM_index
    if ell < abs(self.s) or ell < abs(m):
        raise ValueError(f"Bad index (ell, m)=({ell}, {m}) for spin weight s={self.s}")
    if ell < self.ell_min or ell > self.ell_max:
        raise ValueError(f"Requested ell index {ell} outside bounds of this data ({self.ell_min, self.ell_max})")
    return LM_index(ell, m, self.ell_min)


def truncate_ell(self, new_ell_max):
    """Slice array so that new ell max is no more than the given value

    Note that the result's maximum ell value is limited by the current maximum.  If the input is
    larger than that, the object is returned unchanged.

    """
    if new_ell_max >= self.ell_max:
        return self
    truncated = self[..., :self.index(new_ell_max, new_ell_max)+1]
    truncated._metadata['ell_max'] = new_ell_max
    return truncated


def grid(self, n_theta=None, n_phi=None, **kwargs):
    """Return values of function on an equi-angular grid

    This method uses `spinsfast` to convert mode weights of spin-weighted function to values on a
    grid.  The grid has `n_theta` evenly spaced points along the usual polar (colatitude) angle
    theta, and `n_phi` evenly spaced points along the usual azimuthal angle phi.  This grid
    corresponds to the one produced by `spherical_functions.theta_phi`; see that function for
    specifics.

    The output array has one more dimension than this object; rather than the last axis giving the
    mode weights, the last two axes give the values on the two-dimensional grid.

    Parameters
    ==========
    n_theta: None or int [defaults to None]
        Number of points to use in theta direction.  None is equivalent to 2*self.ell_max+1, which
        is the minimum number that can capture behavior up to and including ell_max.  If you need to
        multiply the result with some `other` spin-weighted function, you should use an n_theta
        value of 2 * (self.ell_max + other.ell_max) + 1 to avoid aliasing.
    n_phi: None or int [defaults to None]
        Number of points to use in the phi direction.  Here, None is equivalent to n_phi=n_theta,
        after calculation of the default value for n_theta.  Note that the same comments apply about
        avoiding aliasing.
    **kwargs: any types
        Additional keyword arguments are passed through to the Grid constructor on output

    """
    import copy
    import numpy as np
    import spinsfast
    from .. import Grid
    n_theta = n_theta or 2*self.ell_max+1
    n_phi = n_phi or n_theta
    metadata = copy.copy(self._metadata)
    metadata.pop('ell_max', None)
    metadata.update(**kwargs)
    return Grid(spinsfast.salm2map(self.view(np.ndarray), self.s, self.ell_max, n_theta, n_phi), **metadata)


def evaluate(self, rotors, **kwargs):
    """Return values of function on input rotors"""
    import numpy as np
    import spherical_functions as sf
    SWSHs = sf.SWSH_grid(rotors, self.spin_weight, self.ell_max)
    return np.tensordot(
        self.view(np.ndarray),
        SWSHs[..., sf.LM_index(self.ell_min, -self.ell_min, 0):sf.LM_index(self.ell_max, self.ell_max, 0)+1],
        axes=([-1], [-1])
    )


def _check_broadcasting(self, array, reverse=False):
    """Test whether or not the given array can broadcast against this object"""
    import numpy as np

    if isinstance(array, type(self)):
        try:
            if reverse:
                np.broadcast(array, self)
            else:
                np.broadcast(self, array)
        except ValueError:
            return False
        else:
            return True
    else:
        if np.ndim(array) > np.ndim(self)-1:
            raise ValueError(
                f"Cannot broadcast array of {np.ndim(array)} dimensions against {type(self).__name__} "
                f"object of fewer ({np.ndim(self)-1}) non-mode dimensions.\n"
                "This is to ensure that scalars do not operate on individual "
                "mode weights; they must operate on all simultaneously.\n"
                "If that is the case and you still want to broadcast, add more "
                "dimensions before this object's first dimension."
            )
        try:
            if reverse:
                np.broadcast(array, self[..., 0])
            else:
                np.broadcast(self[..., 0], array)
        except ValueError:
            return False
        else:
            return True
