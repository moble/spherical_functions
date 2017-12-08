from __future__ import print_function, division, absolute_import

import numpy as np
from numpy import *
import pytest
import spherical_functions as sf

multiplication_functions = []#sf.multiply]

try:
    import spinsfast
    spinsfast_not_present = False

    def spinsfast_multiply(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g):
        """Multiply functions pointwise

        This function takes the same arguments as the spherical_functions.multiply function and
        returns the same quantities.  However, this performs the multiplication simply by
        transforming the modes to function values on a (theta, phi) grid, multiplying those values
        at each point, and then transforming back to modes.

        """
        s_fg = s_f + s_g
        ellmin_fg = 0
        ellmax_fg = ellmax_f + ellmax_g
        n_theta = 2*ellmax_fg + 1
        n_phi = n_theta
        f_map = spinsfast.salm2map(f, s_f, ellmax_f, n_theta, n_phi)
        g_map = spinsfast.salm2map(g, s_g, ellmax_g, n_theta, n_phi)
        fg_map = f_map * g_map
        fg = spinsfast.map2salm(fg_map, s_fg, ellmax_fg)
        return fg, ellmin_fg, ellmax_fg, s_fg

    multiplication_functions += [spinsfast_multiply]

except ImportError:
    spinsfast_not_present = True

requires_spinsfast = pytest.mark.skipif(spinsfast_not_present, reason="Requires spinsfast to be importable")

@pytest.mark.parametrize("multiplication_function", multiplication_functions)
def test_trivial_multiplication(multiplication_function):
    """Test 1*g and g*1

    We test trivial multiplication by 1.  The modes of the `1` function are just sqrt(4*pi) for the
    (ell,m)=(0,0) mode, and 0 for everything else.  Even though this can be described by just the
    first element, we test this for each ellmax_f up to 8 just to make sure the multiplication
    function works as expected.  This is multiplied by a function with random values input for its
    modes.

    """
    np.random.seed(1234)
    atol = 2e-15
    rtol = 2e-15
    ellmax_g = 8
    for ellmax_f in range(1, ellmax_g):
        f = np.zeros(sf.LM_total_size(0, ellmax_f), dtype=np.complex)
        f[0] = np.sqrt(4*np.pi)
        ellmin_f = 0
        s_f = 0
        ellmin_g = 0
        for s_g in range(1-ellmax_g, ellmax_g):
            i_max = sf.LM_index(ellmax_g, ellmax_g, ellmin_g)
            g = np.random.rand(sf.LM_total_size(0, ellmax_g)) + 1j*np.random.rand(sf.LM_total_size(0, ellmax_g))
            g[:sf.LM_total_size(0, abs(s_g))] = 0.0
            fg, ellmin_fg, ellmax_fg, s_fg = multiplication_function(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g)
            assert s_fg == s_f + s_g
            assert ellmin_fg == 0
            assert ellmax_fg == ellmax_f + ellmax_g
            assert np.allclose(fg[:i_max+1], g[:i_max+1], atol=atol, rtol=rtol)
            gf, ellmin_gf, ellmax_gf, s_gf = multiplication_function(g, ellmin_g, ellmax_g, s_g, f, ellmin_f, ellmax_f, s_f)
            assert s_gf == s_g + s_f
            assert ellmin_gf == 0
            assert ellmax_gf == ellmax_g + ellmax_f
            assert np.allclose(gf[:i_max+1], g[:i_max+1], atol=atol, rtol=rtol)


@pytest.mark.parametrize("multiplication_function", multiplication_functions)
def test_first_nontrivial_multiplication(multiplication_function):
    """Test f*g where f is an ell=1, s=0 function

    We can write out the expression for f*g in the case where f is a pure ell=1 function, so we can
    check the multiplication function against that formula.  As in `test_trivial_multiplication`, we
    test with multiple values of ellmax_f, and construct `g` from random values.

    """
    def specialized_multiplication(f, ellmax_f, g, ellmin_g, ellmax_g, s_g):
        s_f = 0
        s_fg = s_f + s_g
        ellmin_fg = 0
        ellmax_fg = ellmax_f + ellmax_g
        fg = np.zeros(sf.LM_total_size(0, ellmax_fg), dtype=np.complex)
        for ell in range(ellmin_fg, ellmax_fg+1):
            for m in range(-ell, ell+1):
                i_fg = sf.LM_index(ell, m, 0)
                if ell+1 <= ellmax_g:
                    if m-1 >= -(ell+1):
                        i_g = sf.LM_index(ell+1, m-1, 0)
                        fg[i_fg] += (
                            np.sqrt((2*ell+3)*(2*ell+1))
                            * sf.Wigner3j(1, ell+1, ell, 0, s_g, s_g)
                            * sf.Wigner3j(1, ell+1, ell, 1, m-1, m)
                            * f[3]
                            * g[i_g]
                        )
                    if m >= -(ell+1) and m <= (ell+1):
                        i_g = sf.LM_index(ell+1, m, 0)
                        fg[i_fg] += (
                            np.sqrt((2*ell+3)*(2*ell+1))
                            * sf.Wigner3j(1, ell+1, ell, 0, s_g, s_g)
                            * sf.Wigner3j(1, ell+1, ell, 0, m, m)
                            * f[2]
                            * g[i_g]
                        )
                    if m+1 <= (ell+1):
                        i_g = sf.LM_index(ell+1, m+1, 0)
                        fg[i_fg] += (
                            np.sqrt((2*ell+3)*(2*ell+1))
                            * sf.Wigner3j(1, ell+1, ell, 0, s_g, s_g)
                            * sf.Wigner3j(1, ell+1, ell, -1, m+1, m)
                            * f[1]
                            * g[i_g]
                        )
                if ell-1 >= ellmin_g and ell-1 >= abs(s_fg):
                    if m-1 >= -(ell-1):
                        i_g = sf.LM_index(ell-1, m-1, 0)
                        fg[i_fg] += (
                            np.sqrt((2*ell-1)*(2*ell+1))
                            * sf.Wigner3j(1, ell-1, ell, 0, s_g, s_g)
                            * sf.Wigner3j(1, ell-1, ell, 1, m-1, m)
                            * f[3]
                            * g[i_g]
                        )
                    if m >= -(ell-1) and m <= (ell-1):
                        i_g = sf.LM_index(ell-1, m, 0)
                        fg[i_fg] += (
                            np.sqrt((2*ell-1)*(2*ell+1))
                            * sf.Wigner3j(1, ell-1, ell, 0, s_g, s_g)
                            * sf.Wigner3j(1, ell-1, ell, 0, m, m)
                            * f[2]
                            * g[i_g]
                        )
                    if m+1 <= (ell-1):
                        i_g = sf.LM_index(ell-1, m+1, 0)
                        fg[i_fg] += (
                            np.sqrt((2*ell-1)*(2*ell+1))
                            * sf.Wigner3j(1, ell-1, ell, 0, s_g, s_g)
                            * sf.Wigner3j(1, ell-1, ell, -1, m+1, m)
                            * f[1]
                            * g[i_g]
                        )
                fg[i_fg] *= np.sqrt(5/(4*np.pi))
        return fg, ellmin_fg, ellmax_fg, s_fg

    
    np.random.seed(1234)
    atol = 2e-10
    rtol = 2e-10
    ellmax_g = 8
    for ellmax_f in range(1, ellmax_g):
        f = np.zeros(sf.LM_total_size(0, ellmax_f), dtype=np.complex)
        f[1:4] = np.random.rand(3) + 1j*np.random.rand(3)
        ellmin_f = 0
        s_f = 0
        ellmin_g = 0
        for s_g in range(1-ellmax_g, ellmax_g):
            i_max = sf.LM_index(ellmax_g, ellmax_g, ellmin_g)
            g = np.random.rand(sf.LM_total_size(0, ellmax_g)) + 1j*np.random.rand(sf.LM_total_size(0, ellmax_g))
            g[:sf.LM_total_size(0, abs(s_g))] = 0.0
            fg, ellmin_fg, ellmax_fg, s_fg = multiplication_function(f, ellmin_f, ellmax_f, s_f, g, ellmin_g, ellmax_g, s_g)
            fg2, ellmin_fg2, ellmax_fg2, s_fg2 = specialized_multiplication(f, ellmax_f, g, ellmin_g, ellmax_g, s_g)
            assert s_fg == s_f + s_g
            assert ellmin_fg == 0
            assert ellmax_fg == ellmax_f + ellmax_g
            assert np.allclose(fg[:i_max+1], fg2[:i_max+1], atol=atol, rtol=rtol)
            gf, ellmin_gf, ellmax_gf, s_gf = multiplication_function(g, ellmin_g, ellmax_g, s_g, f, ellmin_f, ellmax_f, s_f)
            assert s_gf == s_g + s_f
            assert ellmin_gf == 0
            assert ellmax_gf == ellmax_g + ellmax_f
            assert np.allclose(gf[:i_max+1], fg2[:i_max+1], atol=atol, rtol=rtol)
