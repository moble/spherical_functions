# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the SWSHModes class.  In
### particular, they assume that the first argument, `self` is an instance of SWSHModes.  They
### should probably not be used outside of that class.


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

def grid(self, n_theta=None, n_phi=None):
    """Return values of function on an equi-angular grid

    This method uses `spinsfast` to convert mode weights of spin-weighted function to values on a
    grid.  The grid has `n_theta` evenly spaced points along the usual polar (colatitude) angle
    theta, and `n_phi` evenly spaced points along the usual azimuthal angle phi.  This grid
    corresponds to the one produced by `spherical_functions.theta_phi`.  The output array has one
    more dimension than this object; rather than the last axis giving the mode weights, the last two
    axes give the values on the two-dimensional grid.

    """
    import numpy as np
    import spinsfast
    n_theta = n_theta or 2*self.ell_max+1
    n_phi = n_phi or n_theta
    return spinsfast.salm2map(self.view(np.ndarray), self.s, self.ell_max, n_theta, n_phi)

