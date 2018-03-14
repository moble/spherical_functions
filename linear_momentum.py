import numpy as np
import math
from spherical_functions import LM_total_size, LM_index, Wigner3j
from numba import jit

@jit('Tuple((floatc, floatc, floatc))(complex128[:], intc, intc)')
def p_multiply(f, ellmin_f, ellmax_f):
    """Return modes of the decomposition of f*g

    s1Yl1m1 * s2Yl2m2 = sum([
        s3Yl3m3.conjugate() * (-1)**(l1+l2+l3) * sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*pi))
            * Wigner3j(l1, l2, l3, s1, s2, s3) * Wigner3j(l1, l2, l3, m1, m2, m3)
        for l3 in range(abs(l1-l2), l1+l2+1)
    ])

    Here, s3 = -s1 - s2 and m3 = -m1 - m2.

    For general f and g with random ellmin/max and m:
    f*g = sum([ sqrt((2*l1+1)/4*pi)*f(l1,m1)
               sum([ sqrt(2*l2+1)*g(l2,m2)
                    sum([ s3Yl3m3*sqrt(2*l3+1)*(-1)**(l1+l2+l3+s3+m3)
                          *Wigner3j(l1, l2, l3, s1, s2, -s3) * Wigner3j(l1, l2, l3, m1, m2, -m3)
                    for l3 in range(abs(l1-l2), l1+l2+1)
                    ])
               for l2,m2 in range(ellmin_g, ellmax_g)
               ])
          for l1,m1 in range(ellmin_f, ellmax_f)
          ])
    for s3 = s_f+s_g, m3 = m1+m2

    For linear momentum mulitplication, we follow the form in Boyle (2014) in section 4:
    dp/domega dt = R^2/16 pi |dh/dt|^2 r\hat ->
    dp_j/dt = R^2/16 sum([
    r\hat^(1,m'-m)_j* h\dot ^(l,m)* h\dot\bar^(l',m')*(-1)^m'* \sqrt(3(2l+1)(2l'+1)/4 pi)
    *Wigner3j(l, l',1,m,-m',m'-m)*Wigner3j(l,l',1,2,-2,0)
    for l, l', m, m' 
    ])

    where l' runs over (l-1, l, l+1), and m'=(m-1, m+1) for j=x,y and m'=0 for j=z.

    Parameters
    ----------
    f: complex array
        This gives the mode weights of the function `f` expanded in spin-weighted spherical
        harmonics.  They must be stored in the standard order:
            [f(ell, m) for ell in range(ellmin_f, ellmax_f+1) for m in range(-ell, ell+1)]
        In particular, it is permissible to have ellmin < |s|, even though any coefficients for such
        values should be zero; they will just be ignored. Assume s_f = 2.
    ellmin_f: int
        See `f`
    ellmax_f: int
        See `f`
    
    Returns
    -------
    dp_x/dt: float
    dp_y/dt: float
    dp_z/dt: float
    """
    print("__________Incomplete!! Don't use!!_______________")

    for ell1 in range(ellmin_f, ellmax_f+1):
        for m1 in range(-ell1, ell1+1):
            for ell2 in range(ell1-1, ell1+2): #ell2 only has values ell1-1, ell1, ell1+1
                #m2=0 for z componenet
                pz += f[LM_index(ell1,m1,ellmin_f)]*f[LM_index(ell2,m1,ellmin_f)].conjugate()*
                math.sqrt((2*ell1+1)*(2*ell2+1))*Wigner3j(ell1,ell2,1,m,-m,0)*Wigner3j(ell1,ell2,1,2,-2,0)
                
                 #m2 only has values m1-1 and m1+1 for x,y components
                py += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,2,-2,0)*(
                    f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) + 
                    f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                px += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,2,-2,0)*(
                    f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) - 
                    f[LM_index(ell2,m1+1mellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))
            

    return px*pow(R,2)/(16*math.pi),py*complex(0,1)*pow(R,2)/(16*math.pi),pz*pow(R,2)/(16*math.pi)
