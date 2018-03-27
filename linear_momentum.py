import numpy as np
import math
from spherical_functions import LM_total_size, LM_index, Wigner3j
from numba import jit
import matplotlib.pyplot as plt
import argparse
import os
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scri


#From COMCorrections/MikeNotes/integrals2.py
def start_with_ell_equals_zero(data, ell_min=None):
    """Make sure sYlm mode data starts with ell=0
    
    This is helpful to pass the data into spinsfast.
    """
    if isinstance(data, scri.waveform_base.WaveformBase):
        ell_min = data.ell_min
        data = data.data
    if ell_min is None:
        raise ValueError('Input ell_min was None, but data does not describe the ell_min')
    zeros = np.zeros((data.shape[0], LM_total_size(0, abs(ell_min)-1)), dtype=data.dtype)
    # print('zeros.shape =', zeros.shape, '\tzeros.dtype =', zeros.dtype, '\tell_min =', ell_min)
    return np.concatenate((zeros, data), axis=1)

@jit('Tuple((complex128, complex128, complex128))(complex128[:], intc, intc)')
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
        values should be zero; they will just be ignored. Assume s_f = 2, and that f is the time 
        derivative of the metric perturbation h.
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
   
    pz,py,px = 0.0,0.0,0.0 #initialize

    for ell1 in range(ellmin_f, ellmax_f+1):
        for m1 in range(-ell1, ell1+1):
            if ell1==ellmin_f:
                ell2min = ell1
                ell2max = ell1+2
            elif ell1 == ellmax_f:
                ell2min = ell1-1
                ell2max = ell1+1
            else:
                ell2min = ell1-1
                ell2max = ell1+2
            
            for ell2 in range(ell2min, ell2max): #ell2 only has values ell1-1, ell1, ell1+1
                #m2=0 for z component
                pz += (f[LM_index(ell1,m1,ellmin_f)]*f[LM_index(ell2,m1,ellmin_f)].conjugate()*
                       math.sqrt((2*ell1+1)*(2*ell2+1))*Wigner3j(ell1,ell2,1,m1,-m1,0)*Wigner3j(ell1,ell2,1,2,-2,0))
                
                 #m2 only has values m1-1 and m1+1 for x,y components
                if m1 > -ell1 and m1 <ell1-1:
                    py += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) + 
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                    px += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) - 
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                elif m1 == -ell1:
                    py += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                    px -= f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                else:
                    py += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1))

                    px += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1))


    return px/(16*math.pi),py*complex(0,1)/(16*math.pi),pz/(16*math.pi)


def main():
    """Calculate the linear momentum kicks from various runs at the same time, taking 
    waveform data from rhOverM extrapolated data, both COM corrected and original.

    INCOMPLETE: need to line up runs at merger for plotting. 
    """

    parser = argparse.ArgumentParser(
        description = "Calculate and return plots for linear momentum kicks.")
    parser.add_argument("--dir", nargs="+", required = "True",
                        help = "Directory in which to start looking for waveform data")
    parser.add_argument("--filename", required = "True", dest = "filename",
                        help = "Filename root for output files.")

    parser.add_argument("--COM_corrected", action = "store_true",
                        help = "Include if you want to look at the COM corrected data.")

    args = parser.parse_args()

    for directory in args.dir: #Consider all directories given in argument, 
        if args.COM_corrected:
            datadirs = [os.path.join(d,x)
                        for d, dirs, files in os.walk(directory, topdown=True)
                        for x in files if x.endswith("rhOverM_Asymptotic_GeometricUnits_CoM.h5") and os.path.isfile(d+'/'+x)]
    
        else:
            datadirs = [os.path.join(d,x)
                        for d, dirs, files in os.walk(directory, topdown=True)
                        for x in files if x.endswith("rhOverM_Asymptotic_GeometricUnits.h5") and os.path.isfile(d+'/'+x)]

    datadirs = sorted(datadirs)
    datadirs_maxLev = [] #only use the highest resolution available

    for x in range(len(datadirs)-1):
            name1 = datadirs[x]
            name2 = datadirs[x+1]
            if name1[:-16]==name2[:-16]:
                pass
            else:
                datadirs_maxLev.append(name1)
    datadirs_maxLev.append(datadirs[-1])

    #Extrapolations = ['Extrapolated_N2.dir', 'Extrapolated_N3.dir', 'Extrapolated_N4.dir', 
    #                  'OutermostExtraction.dir']
    extrapolation = 'Extrapolated_N4.dir'
    
    p = [] #Holders for linear momentum data, get 3 componenets for each time step
    pmag = [] #p magnitude
    times = [] #holders for the time values for each simulation considered

    for datadir in datadirs_maxLev: #Each directory in the non-COM corrected pool
        #for extrapolation in Extrapolations:
          
        with  open(datadir[:-36]+'metadata.txt') as meta: #get merger time from metadata.txt
            for line in meta:
                if 'common-horizon-time' in line:
                    try:
                        t_comhor = float(line.split()[-1])
                    except ValueError:
                        pass
        
        h = scri.SpEC.read_from_h5(datadir+'/'+extrapolation)
        U = h.t[:]
        times.append(U)
        h = start_with_ell_equals_zero(h)
        hdot = np.empty_like(h)

        p_temp = [] #all momentum values for the current simulation
        pmag_temp = [] #all momentum magnitude values for the current simulation

        for j in range(h.shape[1]):
            hdot[:,j] = (spline(U,h[:,j].real,k=5).derivative()(U) + #real part derived
                1j*spline(U,h[:,j].imag,k=5).derivative()(U))  #imaginary part derived, added back in 

        for hdot_pt in hdot:
            p_temp.append(p_multiply(hdot_pt,2,8)) #Always ellmin=2, ellmax=8
            #Each component of each element of p is complex, so that the modulus of each using abs() to calcualte
            #the overall magnitude. ie. |p| = sqrt(|px|^2 + |py|^2 + |pz|^2)
            pmag_temp.append(math.sqrt(pow(abs(p_temp[-1][0]),2)+pow(abs(p_temp[-1][1]),2)+pow(abs(p_temp[-1][2]),2)))


        p.append(p_temp)
        pmag.append(pmag_temp)

            #Choose color based on extrapolation
            #if 'N2' in extrapolation:
             #   col = 'g'
            #elif 'N3' in extrapolation:
             #   col = 'b'
            #elif 'N4' in extrapolation:
             #   col = 'm'
            #else:
             #   col = 'r'

    colr = 'b'
    coli = 'g'
    for idx in range(len(datadirs_maxLev)):
        plt.plot(times[idx],pmag[idx],color = colr, alpha = 0.7, linewidth = 0.5 )
    plt.title(r'|$\vec{p}$| vs time')
    plt.xlabel(r'$t/M$')
    plt.ylabel(r'$|\vec{p}|$')
    plt.savefig(args.filename+'_pmagvstime.pdf', bbox_inches = "tight")
    plt.clf()

    fig1, (ax1,ax2) = plt.subplots(2,1, sharex = True)

    for idx in range(len(datadirs_maxLev)):
        ax1.plot(times[idx],[item[0].real for item in p[idx]],color = colr,alpha = 0.4, linewidth = 0.5)
        ax2.plot(times[idx],[item[0].imag for item in p[idx]],color = coli, alpha = 0.4, linewidth = 0.5)
    ax1.set_title(r'$p_x$ vs time')
    ax2.set_xlabel(r'$t/M$')
    ax1.set_ylabel(r'Re($p_x$)')
    ax2.set_ylabel(r'Im($p_x$)')
    plt.savefig(args.filename+'_pxvstime.pdf', bbox_inches="tight")
    plt.clf()

    fig2, (ax1,ax2) = plt.subplots(2,1, sharex = True)

    for idx in range(len(datadirs_maxLev)):
        ax1.plot(times[idx],[item[1].real for item in p[idx]],color = colr,alpha = 0.4, linewidth = 0.5)
        ax2.plot(times[idx],[item[1].imag for item in p[idx]],color = coli, alpha = 0.4, linewidth = 0.5)
    ax1.set_title(r'$p_y$ vs time')
    ax2.set_xlabel(r'$t/M$')
    ax1.set_ylabel(r'Re($p_y$)')
    ax2.set_ylabel(r'Im($p_y$)')
    plt.savefig(args.filename+'_pyvstime.pdf', bbox_inches="tight")
    plt.clf()

    fig3, (ax1,ax2) = plt.subplots(2,1, sharex = True)

    for idx in range(len(datadirs_maxLev)):
        ax1.plot(times[idx],[item[2].real for item in p[idx]],color = colr,alpha = 0.4,linewidth = 0.5)
        ax2.plot(times[idx],[item[2].imag for item in p[idx]],color = coli, alpha = 0.4, linewidth = 0.5)
    ax1.set_title(r'$p_z$ vs time')
    ax2.set_xlabel(r'$t/M$')
    ax1.set_ylabel(r'Re($p_z$)')
    ax2.set_ylabel(r'Im($p_z$)')
    plt.savefig(args.filename+'_pzvstime.pdf', bbox_inches="tight")
    plt.clf()
                 
if __name__=="__main__":
    main()
