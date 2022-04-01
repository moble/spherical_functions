import math
import numpy as np
import numba as nb
from . import LM_total_size, LM_index, jit
from .recursions import Wigner3jCalculator


@jit
def _multiplication_helper(f, ellmin_f, ellmax_f, s_f,
                           g, ellmin_g, ellmax_g, s_g,
                           fg, ellmin_fg, ellmax_fg, s_fg):
    # NOTE: This works in-place on fg *and* returns fg for good measure
    s_calculator = Wigner3jCalculator(ellmax_f, ellmax_g)
    m_calculator = Wigner3jCalculator(ellmax_f, ellmax_g)

    for ell1 in range(ellmin_f, ellmax_f+1):
        sqrt1 = math.sqrt((2*ell1+1)/(4*math.pi))
        for m1 in range(-ell1, ell1+1):
            sum1 = sqrt1*f[..., LM_index(ell1, m1, ellmin_f)]  # Calculate f contribution
            for ell2 in range(ellmin_g, ellmax_g+1):
                sqrt2 = math.sqrt(2*ell2+1)
                w3j_s = s_calculator.calculate(ell1, ell2, s_f, s_g)
                for m2 in range(-ell2, ell2+1):
                    w3j_m = m_calculator.calculate(ell1, ell2, m1, m2)
                    sum2 = sqrt2*g[..., LM_index(ell2, m2, ellmin_g)]   # Calculate g contribution
                    m3 = m1+m2
                    for ell3 in range(max(abs(m3), abs(ell1-ell2)), min(ell1+ell2, ellmax_fg)+1):
                        # Could loop over same (ell3, m3) more than once, so add all contributions together
                        fg[..., LM_index(ell3, m3, ellmin_fg)] += (
                            math.pow(-1, ell1 + ell2 + ell3 + s_fg + m3)
                            * math.sqrt(2*ell3+1)
                            * w3j_s[ell3]  # Wigner3j(ell1, ell2, ell3, s_f, s_g, -s_fg)
                            * w3j_m[ell3]  # Wigner3j(ell1, ell2, ell3, m1, m2, -m3)
                        ) * sum1 * sum2
    return fg


def multiply(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g):
    """Return modes of the decomposition of f*g

    s1Yl1m1 * s2Yl2m2 = sum([
        s3Yl3m3.conjugate() * (-1)**(l1+l2+l3) * sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*pi))
            * Wigner3j(l1, l2, l3, s1, s2, s3) * Wigner3j(l1, l2, l3, m1, m2, m3)
        for l3 in range(abs(l1-l2), l1+l2+1)
    ])

    Here, s3 = -s1 - s2 and m3 = -m1 - m2.

    For general f and g with random ellmin/max and m:
    f*g = sum([ sqrt(3*(2*l1+1)/4*pi)*f(l1,m1)
               sum([ sqrt(2*l2+1)*g(l2,m2)
                    sum([ s3Yl3m3.conjugate()*sqrt(2*l3+1)*(-1)**(l1+l2+l3)
                          *Wigner3j(l1, l2, l3, s1, s2, s3) * Wigner3j(l1, l2, l3, m1, m2, m3)
                    for l3 in range(abs(l1-l2), l1+l2+1)
                    ])
               for l2,m2 in range(ellmin_g, ellmax_g)
               ])
          for l1,m1 in range(ellmin_f, ellmax_f)
          ])
    for same s3,m3 as above


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
    shape_fg = np.broadcast(f[..., 0], g[..., 0]).shape + (LM_total_size(0, ellmax_fg),)
    fg = np.zeros(shape_fg, dtype=complex)

    _multiplication_helper(f, ellmin_f, ellmax_f, s_f,
                           g, ellmin_g, ellmax_g, s_g,
                           fg, ellmin_fg, ellmax_fg, s_fg)

    return fg, ellmin_fg, ellmax_fg, s_fg
