#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import math
import cmath
import numpy as np
import quaternion
import spherical_functions as sf
import pytest


def test_modes_creation():
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8

        # Test failed creation with real data
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2)
        with pytest.raises(ValueError):
            m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)

        # Test successful creation with complex data of the right shape
        a = a.view(complex)
        m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
        m = sf.Modes(a, s=s, ell_min=ell_min)
        assert m.s == s
        assert m.ell_min == 0  # NOTE: This is hard coded!!!
        assert m.ell_max == ell_max

        # Test failed creation with complex data of inconsistent shape
        if ell_min != 0:
            with pytest.raises(ValueError):
                m = sf.Modes(a, s=s)
        with pytest.raises(ValueError):
            m = sf.Modes(a, s=s, ell_min=ell_min-1, ell_max=ell_max)
        with pytest.raises(ValueError):
            m = sf.Modes(a, s=s, ell_min=ell_min+1, ell_max=ell_max)
        with pytest.raises(ValueError):
            m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max-1)
        with pytest.raises(ValueError):
            m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max+1)

        # Test failed creation with complex data of impossible shape
        with pytest.raises(ValueError):
            m = sf.Modes(a[..., 1:], s=s, ell_min=ell_min)


def test_modes_grid():
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
        n = 2*ell_max+1
        for n_theta, n_phi in [[None, None], [n, None], [None, n], [n, n], [n+1, n], [n, n+1], [n+1, n+1]]:
            g = m.grid(n_theta=n_theta, n_phi=n_phi)
            assert g.dtype == np.complex
            assert g.shape[:-2] == a.shape[:-1]
            if n_theta is None:
                n_theta = n
            if n_phi is None:
                n_phi = n
            assert g.shape[-2:] == (n_theta, n_phi)


def test_modes_addition():
    tolerance = 1e-14
    for s1 in range(-2, 2 + 1):
        for s2 in range(-s1, s1 + 1):
            ell_min1 = abs(s1)
            ell_max1 = 8
            a1 = np.random.rand(3, 7, sf.LM_total_size(ell_min1, ell_max1)*2).view(complex)
            m1 = sf.Modes(a1, s=s1, ell_min=ell_min1, ell_max=ell_max1)
            ell_min2 = ell_min1+1
            ell_max2 = ell_max1-1
            a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
            m2 = sf.Modes(a2, s=s2, ell_min=ell_min2, ell_max=ell_max2)
            if s1 != s2:
                with pytest.raises(ValueError):
                    m1m2 = m1.subtract(m2)
            else:
                m1m2 = m1.add(m2)
                assert m1m2.s == s1
                assert m1m2.ell_max == m1.ell_max
                i1 = sf.LM_total_size(0, min(ell_min1, ell_min2)-1)
                i2 = sf.LM_total_size(0, max(ell_min1, ell_min2)-1)
                i3 = sf.LM_total_size(0, min(ell_max1, ell_max2))
                i4 = sf.LM_total_size(0, max(ell_max1, ell_max2))
                assert np.array_equiv(m1m2[..., :i1], 0.0)
                assert np.array_equal(m1m2[..., i1:i2], a1[..., :i2-i1])
                assert np.array_equal(m1m2[..., i1:i2], m1.view(np.ndarray)[..., i1:i2])
                assert np.array_equal(m1m2[..., i2:i3], m1.view(np.ndarray)[..., i2:i3]+m2.view(np.ndarray)[..., i2:i3])
                assert np.array_equal(m1m2[..., i3:i4], m1.view(np.ndarray)[..., i3:i4])
                g12 = m1m2.grid()
                n_theta, n_phi = g12.shape[-2:]
                g1 = m1.grid(n_theta, n_phi)
                g2 = m2.grid(n_theta, n_phi)
                assert np.allclose(g1+g2, g12, rtol=tolerance, atol=tolerance)


def test_modes_subtraction():
    tolerance = 1e-14
    np.random.seed(1234)
    for s1 in range(-2, 2 + 1):
        for s2 in range(-s1, s1 + 1):
            ell_min1 = abs(s1)
            ell_max1 = 8
            a1 = np.random.rand(3, 7, sf.LM_total_size(ell_min1, ell_max1)*2).view(complex)
            m1 = sf.Modes(a1, s=s1, ell_min=ell_min1, ell_max=ell_max1)
            ell_min2 = ell_min1+1
            ell_max2 = ell_max1-1
            a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
            m2 = sf.Modes(a2, s=s2, ell_min=ell_min2, ell_max=ell_max2)
            if s1 != s2:
                with pytest.raises(ValueError):
                    m1m2 = m1.subtract(m2)
            else:
                m1m2 = m1.subtract(m2)
                assert m1m2.s == s1
                assert m1m2.ell_max == m1.ell_max
                i1 = sf.LM_total_size(0, min(ell_min1, ell_min2)-1)
                i2 = sf.LM_total_size(0, max(ell_min1, ell_min2)-1)
                i3 = sf.LM_total_size(0, min(ell_max1, ell_max2))
                i4 = sf.LM_total_size(0, max(ell_max1, ell_max2))
                assert np.array_equiv(m1m2[..., :i1], 0.0)
                assert np.array_equal(m1m2[..., i1:i2], a1[..., :i2-i1])
                assert np.array_equal(m1m2[..., i1:i2], m1.view(np.ndarray)[..., i1:i2])
                assert np.array_equal(m1m2[..., i2:i3], m1.view(np.ndarray)[..., i2:i3]-m2.view(np.ndarray)[..., i2:i3])
                assert np.array_equal(m1m2[..., i3:i4], m1.view(np.ndarray)[..., i3:i4])
                g12 = m1m2.grid()
                n_theta, n_phi = g12.shape[-2:]
                g1 = m1.grid(n_theta, n_phi)
                g2 = m2.grid(n_theta, n_phi)
                assert np.allclose(g1-g2, g12, rtol=tolerance, atol=tolerance)


def test_modes_multiplication():
    tolerance = 1e-13
    np.random.seed(1234)
    for s1 in range(-2, 2 + 1):
        for s2 in range(-s1, s1 + 1):
            ell_min1 = abs(s1)
            ell_max1 = 8
            a1 = np.random.rand(3, 7, sf.LM_total_size(ell_min1, ell_max1)*2).view(complex)
            m1 = sf.Modes(a1, s=s1, ell_min=ell_min1, ell_max=ell_max1)
            ell_min2 = ell_min1+1
            ell_max2 = ell_max1-1
            a2 = np.random.rand(3, 7, sf.LM_total_size(ell_min2, ell_max2)*2).view(complex)
            m2 = sf.Modes(a2, s=s2, ell_min=ell_min2, ell_max=ell_max2)
            m1m2 = m1.multiply(m2)
            assert m1m2.s == s1 + s2
            assert m1m2.ell_max == m1.ell_max + m2.ell_max
            g12 = m1m2.grid()
            n_theta, n_phi = g12.shape[-2:]
            g1 = m1.grid(n_theta, n_phi)
            g2 = m2.grid(n_theta, n_phi)
            assert np.allclose(g1*g2, g12, rtol=tolerance, atol=tolerance)


def test_modes_conjugate():
    tolerance = 1e-15
    np.random.seed(1234)
    for inplace in [False, True]:
        for s in range(-2, 2 + 1):
            ell_min = abs(s)
            ell_max = 8
            a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
            m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
            g = m.grid()
            s = m.s
            ell_min = m.ell_min
            ell_max = m.ell_max
            shape = m.shape
            mbar = m.conjugate(inplace)
            gbar = mbar.grid()
            assert s == -mbar.s
            assert ell_min == mbar.ell_min
            assert ell_max == mbar.ell_max
            assert shape == mbar.shape
            assert np.allclose(g, np.conjugate(gbar), rtol=tolerance, atol=tolerance)


def test_modes_real():
    tolerance = 1e-14
    for inplace in [False, True]:
        s = 0
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        # Test success with s==0
        m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
        g = m.grid()
        s = m.s
        ell_min = m.ell_min
        ell_max = m.ell_max
        shape = m.shape
        mreal = m.real(inplace)
        greal = mreal.grid()
        assert s == mreal.s
        assert ell_min == mreal.ell_min
        assert ell_max == mreal.ell_max
        assert shape == mreal.shape
        assert np.allclose(greal, np.real(greal)+0.0j, rtol=tolerance, atol=tolerance)
        assert np.allclose(np.real(g), np.real(greal), rtol=tolerance, atol=tolerance)
        assert np.allclose(np.zeros_like(g, dtype=float), np.imag(greal), rtol=tolerance, atol=tolerance)
        # Test failure with s!=0
        m = sf.Modes(a, s=1, ell_min=ell_min, ell_max=ell_max)
        with pytest.raises(ValueError):
            mreal = m.real(inplace)


@pytest.mark.xfail
def test_modes_eth():
    raise NotImplementedError()


def test_modes_norm():
    tolerance = 1e-15
    for s in range(-2, 2 + 1):
        ell_min = abs(s)
        ell_max = 8
        a = np.random.rand(3, 7, sf.LM_total_size(ell_min, ell_max)*2).view(complex)
        m = sf.Modes(a, s=s, ell_min=ell_min, ell_max=ell_max)
        mmbar = m.multiply(m.conjugate())
        norm = np.sqrt(2*np.sqrt(np.pi) * mmbar[..., 0].view(np.ndarray).real)
        assert np.allclose(norm, m.norm(), rtol=tolerance, atol=tolerance)



@pytest.mark.xfail
def test_modes_ufuncs():
    raise NotImplementedError()
