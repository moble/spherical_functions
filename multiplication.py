import numpy as np
from . import LM_total_size, Wigner3j


def multiply(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g):
    """Return modes of the decomposition of f*g

    s1Yl1m1 * s2Yl2m2 = sum([
        s3Yl3m3.conjugate() * (-1)**(l1+l2+l3) * sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*pi))
            * Wigner3j(l1, l2, l3, s1, s2, s3) * Wigner3j(l1, l2, l3, m1, m2, m3)
        for l3 in range(abs(l1-l2), l1+l2+1)
    ])

    Here, s3 = -s1 - s2 and m3 = -m1 - m2.

    Parameters
    ----------
    f: complex array
        This gives the mode weights of the function `f` expanded in spin-weighted spherical
        harmonics.  They must be stored in the standard order:
            [f(ell, m) for ell in range(ellmin_f, ellmax_f+1) for m in range(-ell, ell+1)]
        In particular, it is permissible to have ellmin < |s|, even though any coefficients for such
        values should be zero; they will just be ignored.
    ellmin_f: int
        See `f`
    ellmax_f: int
        See `f`
    s_f: int
        Spin weight of `f` function
    g: complex array
        As above, but for the second function
    ellmin_g: int
        As above, but for the second function
    ellmax_f: int
        As above, but for the second function
    s_f: int
        As above, but for the second function

    Returns
    -------
    fg: complex array
    ellmin_fg: int
    ellmax_fg: int
    s_fg: int
        These outputs are analogous to the inputs, except for f*g.  The spin weight is necessarily
            s_fg = s_f + s_g
        and the max ell value is necessarily
            ellmax_fg = ellmax_f + ellmax_g
        For simplicity, we always return ellmin_fg=0, even though this may not be necessary.

    """
    s_fg = s_f + s_g
    ellmax_fg = ellmax_f + ellmax_g
    ellmin_fg = 0
    fg = np.zeros(LM_total_size(0, ellmax_fg), dtype=np.complex)

    
    raise NotImplementedError("Do the calculations here!")  # and remove this line


    return fg, ellmin_fg, ellmax_fg, s_fg
