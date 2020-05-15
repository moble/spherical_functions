import numpy as np
import numba

@numba.njit
def complex_powers(z, M):
    """Compute integer powers of z=exp(iθ) recursively

    This algorithm is due to Stoer and Bulirsch in "Introduction to Numerical Analysis" (page 24) —
    with a little help from de Moivre's formula, which is essentially exp(iθ)ⁿ = exp(inθ), as well
    as my own alterations to deal with different behaviors in different quadrants.

    There isn't usually a huge advantage to using this specialized function.  If you just need a
    particular power, it will generally be far more efficient and just as accurate to compute either
    exp(iθ)ⁿ or exp(inθ) explicitly.  However, if you need all powers from 0 to M, this function is
    about 10 or 5 times faster than those options, respectively.  Like those options, this function
    is numerically stable, in the sense that its errors are usually smaller than the error from
    machine-precision errors in the input argument — or at worst about 30% larger around π/2.

    Parameters
    ==========
    z: complex
        Complex number to take integer powers of, normalized
    M: int
        Highest power to compute

    Returns
    =======
    zpowers: array of complex
        Powers ẑ⁰, ẑ¹, ..., ẑᴹ (with ẑ = z / |z|)

    """
    z /= np.abs(z)
    θ = 1
    while z.real<0 or z.imag<0:
        θ *= 1j
        z /= 1j
    clock = θ
    zpowers = np.empty(M+1, dtype=np.complex128)
    zpowers[0] = 1.0 + 0.0j
    zpowers[1] = z
    dc = -2 * np.sqrt(z).imag ** 2
    dz = dc + 1j * np.sqrt(-dc * (2 + dc))
    t = 2 * dc
    dz += t * zpowers[1]
    for m in range(2, M+1):
        zpowers[m] = zpowers[m-1] + dz
        dz += t * zpowers[m]
        zpowers[m-1] *= clock
        clock *= θ
    zpowers[M] *= clock
    return zpowers
