import math
import numpy as np
import numba


@numba.njit("float64(int32, int32, int32, int32)")
def A(j, j2, j3, m1):
    return math.sqrt((j**2 - (j2 - j3)**2) * ((j2 + j3 + 1)**2 - j**2) * (j**2 - m1**2))


@numba.njit("int32(int32, int32, int32, int32, int32)")
def B(j, j2, j3, m2, m3):
    return (2 * j + 1) * ((m2 + m3) * (j2 * (j2 + 1) - j3 * (j3 + 1)) - (m2 - m3) * j * (j + 1))


@numba.njit("float64(int32, int32, int32, int32)")
def Xf(j, j2, j3, m1):
    return j * A(j+1, j2, j3, m1)


Yf = B


@numba.njit("float64(int32, int32, int32, int32)")
def Zf(j, j2, j3, m1):
    return (j+1) * A(j, j2, j3, m1)


@numba.njit("void(float64[:], int32, int32)")
def normalize(f, j_min, j_max):
    norm = 0.0
    for j in range(j_min, j_max+1):
        norm += (2 * j + 1) * f[j] ** 2
    f[j_min:j_max + 1] /= math.sqrt(norm)


@numba.njit("void(float64[:], int32, int32, int32, int32, int32, int32)")
def determine_signs(f, j_min, j_max, j2, j3, m2, m3):
    if (f[j_max] < 0.0 and (-1)**(j2-j3+m2+m3) > 0) or (f[j_max] > 0.0 and (-1)**(j2-j3+m2+m3) < 0):
        f[j_min:j_max+1] *= -1.0


@numba.jitclass([
    ('_size', numba.int32),
    ('workspace', numba.float64[:]),
])
class Wigner3jCalculator(object):
    def __init__(self, j2_max, j3_max):
        self._size = j2_max + j3_max + 1
        self.workspace = np.empty(4 * self._size, dtype=np.float64)

    @property
    def size(self):
        return self._size

    def calclulate(self, j2, j3, m2, m3):
        """Compute Wigner 3-j symbols

        For given values of j₂, j₃, m₂, m₃, this computes all valid values of

            ⎛j₁  j₂  j₃⎞   _   ⎛j₂  j₃  j₁⎞
            ⎝m₁  m₂  m₃⎠   -   ⎝m₂  m₃  m₁⎠

        The valid values have m₁=-m₂-m₃ and j₁ ranging from max(|j₂-j₃|, |m₁|) to j₂+j₃.
        The calculation uses the approach described by Luscombe and Luban (1998)
        <https://doi.org/10.1103/PhysRevE.57.7274>, which is a recurrence method,
        leading to significant gains in speed and accuracy.

        The returned array is a slice of this object's `workspace` array, and so will not
        remain the same between calls to this function.  If you want to keep a copy of
        the results, explicitly call the `numpy.copy` method.

        The returned array is indexed by j₁.  In particular, note that some invalid
        values are accessible, but have value 0.

        """
        m1 = -(m2 + m3)

        undefined_min = False
        undefined_max = False
        scale_factor = 1_000.0

        # Set up the workspace
        self.workspace[:] = 0.0
        f = self.workspace[:self.size]
        sf = self.workspace[self.size:2*self.size]
        rf = sf  # An alias for the same memory as `sf`
        F_minus = self.workspace[2*self.size:3*self.size]
        F_plus = self.workspace[3*self.size:4*self.size]

        # Calculate some useful bounds
        j_min = max(abs(j2-j3), abs(m2+m3))
        j_max = j2 + j3

        # Skip all calculations if there are no nonzero elements
        if abs(m2) > j2 or abs(m3) > j3:
            return f
        if j_max < j_min:
            return f

        # When only one term is present, we have a simple formula
        if j_max == j_min:
            f[j_min] = 1.0 / math.sqrt(2.0 * j_min + 1.0)
            if (f[j_min] < 0.0 and (-1) ** (j2-j3+m2+m3) > 0) or (f[j_min] > 0.0 and (-1) ** (j2-j3+m2+m3) < 0):
                f[j_min] *= -1.0
            return f

        ##########################################################################
        # Forward iteration over first "nonclassical" region j_min ≤ j ≤ j_minus #
        ##########################################################################
        Xf_j_min = Xf(j_min, j2, j3, m1)
        Yf_j_min = Yf(j_min, j2, j3, m2, m3)

        if m1 == 0 and m2 == 0 and m3 == 0:
            # Recurrence is undefined, but it's okay because all odd terms must be zero
            F_minus[j_min] = 1.0
            F_minus[j_min+1] = 0.0
            j_minus = j_min + 1

        elif Yf_j_min == 0.0:
            # The second term is either undefined or zero
            if Xf_j_min == 0.0:
                undefined_min = True
                j_minus = j_min
            else:
                F_minus[j_min] = 1.0
                F_minus[j_min+1] = 0.0
                j_minus = j_min + 1

        elif Xf_j_min * Yf_j_min >= 0.0:
            # The second term is already in the classical region
            F_minus[j_min] = 1.0
            F_minus[j_min+1] = -Yf_j_min / Xf_j_min
            j_minus = j_min + 1

        else:
            # Use recurrence relation, Eq. (3) from Luscombe and Luban (1998)
            sf[j_min] = -Xf_j_min / Yf_j_min
            j_minus = j_max
            for j in range(j_min+1, j_max):
                denominator = Yf(j, j2, j3, m2, m3) + Zf(j, j2, j3, m1) * sf[j - 1]
                Xf_j = Xf(j, j2, j3, m1)
                if abs(Xf_j) > abs(denominator) or Xf_j * denominator >= 0.0 or denominator == 0.0:
                    j_minus = j - 1
                    break
                else:
                    sf[j] = -Xf_j / denominator
            F_minus[j_minus] = 1.0
            for k in range(1, j_minus - j_min + 1):
                F_minus[j_minus - k] = F_minus[j_minus - k + 1] * sf[j_minus - k]
            if j_minus == j_min:
                # Calculate at least two terms so that these can be used in three-term recursion
                F_minus[j_min+1] = -Yf_j_min / Xf_j_min
                j_minus = j_min + 1

        if j_minus == j_max:
            # We're finished!
            f[j_min:j_max + 1] = F_minus[j_min:j_max + 1]
            normalize(f, j_min, j_max)
            determine_signs(f, j_min, j_max, j2, j3, m2, m3)
            return f

        ##########################################################################
        # Reverse iteration over second "nonclassical" region j_plus ≤ j ≤ j_max #
        ##########################################################################
        Yf_j_max = Yf(j_max, j2, j3, m2, m3)
        Zf_j_max = Zf(j_max, j2, j3, m1)

        if m1 == 0 and m2 == 0 and m3 == 0:
            # Recurrence is undefined, but it's okay because all odd terms must be zero
            F_plus[j_max] = 1.0
            F_plus[j_max-1] = 0.0
            j_plus = j_max - 1

        elif Yf_j_max == 0.0:
            # The second term is either undefined or zero
            if Zf_j_max == 0.0:
                undefined_max = True
                j_plus = j_max
            else:
                F_plus[j_max] = 1.0
                F_plus[j_max-1] = - Yf_j_max / Zf_j_max
                j_plus = j_max - 1

        elif Yf_j_max * Zf_j_max >= 0.0:
            # The second term is already in the classical region
            F_plus[j_max] = 1.0
            F_plus[j_max-1] = - Yf_j_max / Zf_j_max
            j_plus = j_max - 1

        else:
            # Use recurrence relation, Eq. (2) from Luscombe and Luban (1998)
            rf[j_max] = -Zf_j_max / Yf_j_max
            j_plus = j_min
            for j in range(j_max-1, j_minus - 1, -1):
                denominator = Yf(j, j2, j3, m2, m3) + Xf(j, j2, j3, m1) * rf[j + 1]
                Zf_j = Zf(j, j2, j3, m1)
                if denominator == 0.0 or abs(Zf_j) > abs(denominator) or Zf_j * denominator >= 0.0:
                    j_plus = j + 1
                    break
                else:
                    rf[j] = -Zf_j / denominator
            F_plus[j_plus] = 1.0
            for k in range(1, j_max - j_plus + 1):
                F_plus[j_plus + k] = F_plus[j_plus + k - 1] * rf[j_plus + k]
            if j_plus == j_max:
                F_plus[j_max-1] = - Yf_j_max / Zf_j_max
                j_plus = j_max - 1

        ######################################################################
        # Three-term recurrence over "classical" region j_minus ≤ j ≤ j_plus #
        ######################################################################
        if undefined_min and undefined_max:
            raise ValueError("Cannot initialize recurrence in Wigner3jCalculator.calculate")

        if not undefined_min and not undefined_max:
            # Iterate upwards and downwards, meeting in the middle
            j_mid = (j_minus + j_plus) // 2
            for j in range(j_minus, j_mid):
                F_minus[j+1] = (
                    - (Yf(j, j2, j3, m2, m3) * F_minus[j] + Zf(j, j2, j3, m1) * F_minus[j-1]) / Xf(j, j2, j3, m1)
                )
                if abs(F_minus[j+1]) > 1.0:
                    F_minus[j_min:j+1+1] /= scale_factor
                if abs(F_minus[j+1] / F_minus[j-1]) < 1.0 and F_minus[j+1] != 0.0:
                    j_mid = j + 1
                    break
            F_minus_j_mid = F_minus[j_mid]
            if F_minus[j_mid - 1] != 0.0 and abs(F_minus_j_mid / F_minus[j_mid - 1]) < 1.0e-6:
                j_mid -= 1
                F_minus_j_mid = F_minus[j_mid]
            for j in range(j_plus, j_mid, -1):
                F_plus[j-1] = (
                    - (Xf(j, j2, j3, m1) * F_plus[j+1] + Yf(j, j2, j3, m2, m3) * F_plus[j]) / Zf(j, j2, j3, m1)
                )
                if abs(F_plus[j-1]) > 1.0:
                    F_plus[j-1:j_max+1] /= scale_factor
            F_plus_j_mid = F_plus[j_mid]
            if j_mid == j_max:
                f[j_min:j_max + 1] = F_minus[j_min:j_max + 1]
            elif j_mid == j_min:
                f[j_min:j_max + 1] = F_plus[j_min:j_max + 1]
            else:
                f[j_min:j_mid + 1] = F_minus[j_min:j_mid + 1] * F_plus_j_mid / F_minus_j_mid
                f[j_mid + 1:j_max + 1] = F_plus[j_mid + 1:j_max + 1]

        elif not undefined_min and undefined_max:
            # Iterate upwards only
            for j in range(j_minus, j_plus):
                F_minus[j + 1] = (
                    - (Zf(j, j2, j3, m1) * F_minus[j - 1] + Yf(j, j2, j3, m2, m3) * F_minus[j]) / Xf(j, j2, j3, m1)
                )
                if abs(F_minus[j + 1]) > 1:
                    F_minus[j_min:j+1+1] /= scale_factor
            f[j_min:j_max + 1] = F_minus[j_min:j_max + 1]

        elif undefined_min and not undefined_max:
            # Iterate downwards only
            for j in range(j_plus, j_min, -1):
                F_plus[j-1] = (
                    - (Xf(j, j2, j3, m1) * F_plus[j+1] + Yf(j, j2, j3, m2, m3) * F_plus[j]) / Zf(j, j2, j3, m1)
                )
                if abs(F_plus[j-1]) > 1:
                    F_plus[j-1:j_max+1] /= scale_factor
            f[j_min:j_max + 1] = F_plus[j_min:j_max + 1]

        #############
        # Finish up #
        #############
        normalize(f, j_min, j_max)
        determine_signs(f, j_min, j_max, j2, j3, m2, m3)
        return f
