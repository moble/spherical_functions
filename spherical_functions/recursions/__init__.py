
@numba.njit
def quaternion_angles(R):
    """Compute complex angles for use in Wigner's 𝔇 matrices

    Assuming the Euler angle convention from the quaternions package, we can define

        zᵦ ≔ exp[iβ]
        zₚ ≔ exp[i(α+γ)/2]
        zₘ ≔ exp[i(α-γ)/2]

    It so happens that these combinations can be calculated algebraically from components of a
    quaternion, and are important terms in computing Wigner's 𝔇 matrices.

    """
    a = R[0]**2 + R[3]**2
    b = R[1]**2 + R[2]**2
    sqrta = np.sqrt(a)
    sqrtb = np.sqrt(b)
    zᵦ = ((a - b) + 2j * sqrta * sqrtb) / (a + b)  # exp[iβ]
    zₚ = (R[0] + 1j * R[3]) / sqrta  # exp[i(α+γ)/2]
    zₘ = (R[2] - 1j * R[1]) / sqrtb  # exp[i(α-γ)/2]
    return zᵦ, zₚ, zₘ


def rotate(modes, R):
    """Rotate Modes object by rotor(s)

    Compute fₗₘ = Σₙ fₗₙ 𝔇ˡₙₘ(R), where f is a (possibly spin-weighted) function, fₗₙ are its mode
    weights in the current frame, and fₗₘ are its mode weights in the rotated frame.

    fₗₘ = Σₙ fₗₙ 𝔇ˡₙₘ(R)
         = Σₙ fₗₙ dˡₙₘ(R) exp[iϕₐ(m-n)+iϕₛ(m+n)]
         = Σₙ fₗₙ dˡₙₘ(R) exp[i(ϕₛ+ϕₐ)m+i(ϕₛ-ϕₐ)n]
         = exp[i(ϕₛ+ϕₐ)m] Σₙ fₗₙ dˡₙₘ(R) exp[i(ϕₛ-ϕₐ)n]
         = zₚᵐ Σₙ fₗₙ dˡₙₘ(R) zₘⁿ
         = zₚᵐ {fₗ₀ dˡ₀ₘ(R) + Σₚₙ [fₗₙ dˡₙₘ(R) zₘⁿ + fₗ₋ₙ dˡ₋ₙₘ(R) / zₘⁿ]}
         = zₚᵐ {fₗ₀ ϵ₋ₘ Hˡ₀ₘ(R) + Σₚₙ [fₗₙ ϵₙ ϵ₋ₘ Hˡₙₘ(R) zₘⁿ + fₗ₋ₙ ϵ₋ₙ ϵ₋ₘ Hˡ₋ₙₘ(R) / zₘⁿ]}
         = ϵ₋ₘ zₚᵐ {fₗ₀ Hˡ₀ₘ(R) + Σₚₙ [fₗₙ (-1)ⁿ Hˡₙₘ(R) zₘⁿ + fₗ₋ₙ Hˡ₋ₙₘ(R) / zₘⁿ]}

    Here, n ranges over [-l, l] and pn ranges over [1, l].

    Parameters
    ==========
    modes: Modes
        SWSH modes to rotate
    R: quaternion or array of quaternions
        If this is an array, its shape must be R.shape == modes.shape[:-1]

    """
    fₗₙ = modes.reshape((np.prod(modes.shape[:-1], dtype=int), modes.shape[-1]))
    fₗₘ = np.zeros_like(modes)
    ell_min = modes.ell_min
    ell_max = modes.ell_max

    if not isinstance(R, quaternion) and R.shape != modes.shape[:-1]:
        raise ValueError(
            f"Input rotor must be either a single quaternion or an array with\n"
            f"shape R.shape={R.shape} == modes.shape[:-1]={modes.shape[:-1]}"
        )

    rotors = quaternion.as_float_array(R).reshape((-1, 4))

#unfinished:
    first_slice = slice(None) if rotors.shape[0]==1 else ?

    for iᵣ in range(rotors.shape[0]):
        zᵦ, zₚ, zₘ = quaternion_angles(rotors[i])

        # Compute H elements (basically Wigner's d functions)
#unfinished:
        Hˡₙₘ = H.evaluate(zᵦ)

        # Pre-compute zₚᵐ=exp[i(ϕₛ+ϕₐ)m] for all values of m
        zₚᵐ = complex_powers(zₚ, ell_max)

        for ell in range(ell_min, ell_max+1):
            for m in range(-ell_max, ell_max+1):
                # fₗₘ = ϵ₋ₘ zₚᵐ {fₗ₀ Hˡ₀ₘ(R) + Σₚₙ [fₗ₋ₙ Hˡ₋ₙₘ(R) / zₘⁿ + fₗₙ (-1)ⁿ Hˡₙₘ(R) zₘⁿ]}
                iₘ = fₗₘ.index(ell, m)

                # Initialize with n=0 term
                fₗₘ[first_slice, iₘ] = fₗₙ[fₗₙ.index(ell, 0)] * Hˡₙₘ.element(ell, 0, m)

                # Compute dˡₙₘ terms recursively for 0<n<l, using symmetries for negative n, and
                # simultaneously add the mode weights times zₘⁿ=exp[i(ϕₛ-ϕₐ)n] to the result using
                # Horner form
                negative_terms[first_slice] = fₗₙ[first_slice, fₗₙ.index(ell, -ell)] * Hˡₙₘ.element(ell, -ell, m)
                positive_terms[first_slice] = fₗₙ[first_slice, fₗₙ.index(ell, ell)] * Hˡₙₘ.element(ell, ell, m) * (-1)**ell
                for n in range(ell-1, 0, -1):
                    negative_terms /= zₘ
                    negative_terms += fₗₙ[first_slice, fₗₙ.index(ell, -n)] * Hˡₙₘ.element(ell, -n, m)
                    positive_terms *= zₘ
                    positive_terms += fₗₙ[first_slice, fₗₙ.index(ell, n)] * Hˡₙₘ.element(ell, n, m) * (-1)**n
                fₗₘ[first_slice, iₘ] += negative_terms / zₘ
                fₗₘ[first_slice, iₘ] += positive_terms * zₘ

                # Finish calculation of fₗₘ by multiplying by zₚᵐ=exp[i(ϕₛ+ϕₐ)m]
                fₗₘ[first_slice, iₘ] *= ϵ(-m) * zₚᵐ[m]

    fₗₘ = fₗₘ.reshape(modes.shape)
    return fₗₘ
