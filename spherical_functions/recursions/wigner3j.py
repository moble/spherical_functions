import math
import numpy as np
import numba


@numba.njit("float64(int32, int32, int32, int32)")
def a(j, j2, j3, m1):
    return math.sqrt((j**2 - (j2 - j3)**2) * ((j2 + j3 + 1)**2 - j**2) * (j**2 - m1**2))


@numba.njit("int32(int32, int32, int32, int32, int32, int32)")
def y(j, j2, j3, m1, m2, m3):
    return -(2 * j + 1) * (m1 * (j2 * (j2 + 1)  - j3 * (j3 + 1)) - (m3 - m2) * j * (j + 1))


@numba.njit("float64(int32, int32, int32, int32)")
def x(j, j2, j3, m1):
    return j * a(j+1, j2, j3, m1)


@numba.njit("float64(int32, int32, int32, int32)")
def z(j, j2, j3, m1):
    return (j+1) * a(j, j2, j3, m1)


@numba.njit("void(float64[:], int32, int32)")
def normalize(w3j, jmin, jmax):
    norm = 0.0
    for j in range(jmin, jmax+1):
        norm += (2 * j + 1) * w3j[j]**2
    w3j[jmin:jmax+1] /= math.sqrt(norm)


@numba.njit("void(float64[:], int32, int32, int32, int32, int32, int32)")
def fix_signs(w3j, jmin, jmax, j2, j3, m2, m3):
    if ( (w3j[jmax] < 0.0 and (-1)**(j2-j3+m2+m3) > 0) or (w3j[jmax] > 0.0 and (-1)**(j2-j3+m2+m3) < 0) ):
        w3j[jmin:jmax+1] *= -1.0


@numba.jitclass([
    ('_size', numba.int32),
    ('workspace', numba.float64[:]),
])
class Wigner3jCalculator(object):
    def __init__(self, ell2_max, ell3_max):
        self._size = ell2_max + ell3_max + 1
        self.workspace = np.empty(4 * self._size, dtype=np.float64)

    @property
    def size(self):
        return self._size

    def calclulate(self, j2, j3, m2, m3):
        m1 = -(m2 + m3)
        condition1 = False
        condition2 = False
        scale_factor = 1_000.0

        # Set up the workspace
        self.workspace[:] = 0.0
        w3j = self.workspace[:self.size]
        rs = self.workspace[self.size:2*self.size]
        wl = self.workspace[2*self.size:3*self.size]
        wu = self.workspace[3*self.size:4*self.size]

        # Calculate some useful bounds
        jmin = max(abs(j2-j3), abs(m1))
        jmax = j2 + j3

        # Skip all calculations if there are no nonzero elements
        if abs(m2) > j2 or abs(m3) > j3:
            return w3j
        if jmax < jmin:
            return w3j

        # When only one term is present, we have a simple formula
        if jmax == jmin:
            w3j[jmin] = 1.0 / math.sqrt(2.0 * jmin + 1.0)
            if (w3j[jmin] < 0.0 and (-1)**(j2-j3+m2+m3) > 0) or (w3j[jmin] > 0.0 and (-1)**(j2-j3+m2+m3) < 0):
                w3j[jmin] *= -1.0
            return w3j

        ###########
        # Stage 1 #
        ###########
        xjmin = x(jmin, j2, j3, m1)
        yjmin = y(jmin, j2, j3, m1, m2, m3)

        if m1 == 0 and m2 == 0 and m3 == 0:  # All m's are zero
            wl[jmin] = 1.0
            wl[jmin+1] = 0.0
            jn = jmin + 1

        elif yjmin == 0.0:  # The second term is either zero or undefined
            if xjmin == 0.0:
                condition1 = True
                jn = jmin
            else:
                wl[jmin] = 1.0
                wl[jmin+1] = 0.0
                jn = jmin + 1

        elif xjmin * yjmin >= 0.0:
            # The second term is outside of the non-classical region
            wl[jmin] = 1.0
            wl[jmin+1] = -yjmin / xjmin
            jn = jmin + 1

        else:
            # Calculate terms in the non-classical region
            rs[jmin] = -xjmin / yjmin
            jn = jmax
            for j in range(jmin+1, jmax):
                denom = y(j, j2, j3, m1, m2, m3) + z(j, j2, j3, m1) * rs[j-1]
                xj = x(j, j2, j3, m1)
                if abs(xj) > abs(denom) or xj * denom >= 0.0 or denom == 0.0:
                    jn = j - 1
                    break
                else:
                    rs[j] = -xj / denom
            wl[jn] = 1.0
            for k in range(1, jn - jmin + 1):
                wl[jn-k] = wl[jn-k+1] * rs[jn-k]
            if jn == jmin:
                # Calculate at least two terms so that these can be used in three-term recursion
                wl[jmin+1] = -yjmin / xjmin
                jn = jmin + 1

        if jn == jmax:
            # We're finished!
            w3j[jmin:jmax+1] = wl[jmin:jmax+1]
            normalize(w3j, jmin, jmax)
            fix_signs(w3j, jmin, jmax, j2, j3, m2, m3)
            return w3j

        ###########
        # Stage 2 #
        ###########
        yjmax = y(jmax, j2, j3, m1, m2, m3)
        zjmax = z(jmax, j2, j3, m1)

        if m1 == 0 and m2 == 0 and m3 == 0:
            wu[jmax] = 1.0
            wu[jmax-1] = 0.0
            jp = jmax - 1

        elif yjmax == 0.0:
            if zjmax == 0.0:
                condition2 = True
                jp = jmax
            else:
                wu[jmax] = 1.0
                wu[jmax-1] = - yjmax / zjmax
                jp = jmax-1

        elif yjmax * zjmax >= 0.0:
            wu[jmax] = 1.0
            wu[jmax-1] = - yjmax / zjmax
            jp = jmax - 1

        else:
            rs[jmax] = -zjmax / yjmax
            jp = jmin
            for j in range(jmax-1, jn-1, -1):
                denom = y(j, j2, j3, m1, m2, m3) + x(j, j2, j3, m1) * rs[j+1]
                zj = z(j, j2, j3, m1)
                if abs(zj) > abs(denom) or zj * denom >= 0.0 or  denom == 0.0:
                    jp = j + 1
                    break
                else:
                    rs[j] = -zj / denom
            wu[jp] = 1.0
            for k in range(1, jmax - jp + 1):
                wu[jp+k] = wu[jp+k-1] * rs[jp+k]
            if jp == jmax:
                wu[jmax-1] = - yjmax / zjmax
                jp = jmax - 1

        ###########
        # Stage 3 #
        ###########
        if condition1 and condition2:
            raise ValueError("Invalid input values for Wigner3jCalculator.calclulate")

        if not condition1:
            jmid = (jn + jp) // 2
            for j in range(jn, jmid):
                wl[j+1] = - (z(j, j2, j3, m1) * wl[j-1] + y(j, j2, j3, m1, m2, m3) * wl[j]) / x(j, j2, j3, m1)
                if abs(wl[j+1]) > 1.0:
                    wl[jmin:j+1+1] /= scale_factor
                if abs(wl[j+1] / wl[j-1]) < 1.0 and wl[j+1] != 0.0:
                    jmid = j + 1
                    break
            wnmid = wl[jmid]
            if wl[jmid-1] != 0.0 and abs(wnmid / wl[jmid-1]) < 1.0e-6:
                wnmid = wl[jmid-1]
                jmid -= 1
            for j in range(jp, jmid, -1):
                wu[j-1] = - (x(j, j2, j3, m1) * wu[j+1] + y(j, j2, j3, m1, m2, m3) * wu[j] ) / z(j, j2, j3, m1)
                if abs(wu[j-1]) > 1.0:
                    wu[j-1:jmax+1] /= scale_factor
            wpmid = wu[jmid]
            if jmid == jmax:
                w3j[jmin:jmax+1] = wl[jmin:jmax+1]
            elif jmid == jmin:
                w3j[jmin:jmax+1] = wu[jmin:jmax+1]
            else:
                w3j[jmin:jmid+1] = wl[jmin:jmid+1] * wpmid / wnmid
                w3j[jmid+1:jmax+1] = wu[jmid+1:jmax+1]

        elif condition1 and not condition2:
            for j in range(jp, jmin, -1):
                wu[j-1] = - (x(j, j2, j3, m1) * wu[j+1] + y(j, j2, j3, m1, m2, m3) * wu[j] ) / z(j, j2, j3, m1)
                if abs(wu[j-1]) > 1:
                    wu[j-1:jmax+1] /= scale_factor
            w3j[jmin:jmax+1] = wu[jmin:jmax+1]

        elif condition2 and not condition1:
            for j in range(jn, jp):
                wl[j+1] = - (z(j, j2, j3, m1) * wl[j-1] + y(j, j2, j3, m1, m2, m3) * wl[j]) / x(j, j2, j3, m1)
                if abs(wl[j+1]) > 1:
                    wl[jmin:j+1+1] /= scale_factor
            w3j[jmin:jmax+1] = wl[jmin:jmax+1]

        #############
        # Finish up #
        #############
        normalize(w3j, jmin, jmax)
        fix_signs(w3j, jmin, jmax, j2, j3, m2, m3)
        return w3j
