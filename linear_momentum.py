import numpy as np
import math
from spherical_functions import LM_total_size, LM_index, Wigner3j
from numba import jit
import matplotlib.pyplot as plt
import argparse
import os
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import splrep,splev
from scipy.integrate import simps
import scipy.signal as signal
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

@jit('Tuple((float64, float64, float64))(complex128[:], intc, intc)')
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
            if m1==0 and ell1 > 2:
                continue #skip m=0 modes, except for ell=2,m=0 which should be stable 

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
                pz += (-1)**m1 *(f[LM_index(ell1,m1,ellmin_f)]*f[LM_index(ell2,m1,ellmin_f)].conjugate()*
                       math.sqrt((2*ell1+1)*(2*ell2+1))*Wigner3j(ell1,ell2,1,m1,-m1,0)*Wigner3j(ell1,ell2,1,2,-2,0))
                
                 #m2 only has values m1-1 and m1+1 for x,y components
                if m1 > -ell1 and m1 <ell1:
                    if m1+1 ==0 and ell2 > 2: #don't consider m=0 componenets except for ell=2,m=0
                        py += 1j*(-1)**(m1-1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                            f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1)) 
                        
                        px += (-1)**(m1-1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                            f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1)) 
                             
                    elif m1-1==0 and ell2 > 2: #don't consider m=0 components except for ell=2,m=0
                        py += 1j*(-1)**(m1+1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                            f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))
             
                        px += (-1)**(m1+1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                            -f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))
                    else:
                        py += 1j*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                            (-1)**(m1-1)*f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) +
                            (-1)**(m1+1)*f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                        px += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                            (-1)**(m1-1)*f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) -
                            (-1)**(m1+1)*f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                elif m1 == -ell1:
                    py += 1j*(-1)**(m1+1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                    px -= (-1)**(m1+1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                else:
                    py += 1j*(-1)**(m1-1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1))

                    px += (-1)**(m1-1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1))


    return px.real/(16*math.pi),py.real/(16*math.pi),pz.real/(16*math.pi)

@jit('Tuple((float64, float64, float64))(complex128[:], intc, intc)')
def p_multiply_m0(f, ellmin_f, ellmax_f):
    #same as p_multiply but including m=0 components
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
                pz += (-1)**m1 *(f[LM_index(ell1,m1,ellmin_f)]*f[LM_index(ell2,m1,ellmin_f)].conjugate()*
                                 math.sqrt((2*ell1+1)*(2*ell2+1))*Wigner3j(ell1,ell2,1,m1,-m1,0)*Wigner3j(ell1,ell2,1,2,-2,0))

                 #m2 only has values m1-1 and m1+1 for x,y components                                                                             
                if m1 > -ell1 and m1 <ell1: #m1 +1 and m1-1 are both valid m values
                    py += 1j*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        (-1)**(m1-1)*f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) +
                        (-1)**(m1+1)*f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))
                    
                    px += f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        (-1)**(m1-1)*f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1) -
                        (-1)**(m1+1)*f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))
                    
                elif m1 == -ell1: #m1-1 is not valid
                    py += 1j*(-1)**(m1+1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))
                    
                    px -= (-1)**(m1+1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1+1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,-1-m1,1))

                else: #only other option is m1==ell1, which means m1+1 is not valid
                    py += 1j*(-1)**(m1-1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1))
                    
                    px += (-1)**(m1-1)*f[LM_index(ell1,m1,ellmin_f)]*math.sqrt((2*ell1+1)*(2*ell2+1)/2)*Wigner3j(ell1,ell2,1,2,-2,0)*(
                        f[LM_index(ell2,m1-1,ellmin_f)].conjugate()*Wigner3j(ell1,ell2,1,m1,1-m1,-1))
                    

    return px.real/(16*math.pi),py.real/(16*math.pi),pz.real/(16*math.pi)


def main():
    """Calculate the linear momentum kicks from various runs at the same time, taking 
    waveform data from rhOverM extrapolated data, both COM corrected and original. 
    """

    parser = argparse.ArgumentParser(
        description = "Calculate and return plots for linear momentum kicks.")
    parser.add_argument("--input", nargs="+", required = "True",
                        help = "File in which to find directories to start looking for data")
    parser.add_argument("--filename", required = "True", dest = "filename",
                        help = "Filename root for output files.")

    parser.add_argument("--COM_corrected", action = "store_true",
                        help = "Include if you want to look at the COM corrected data.")

    args = parser.parse_args()

    if args.COM_corrected: #Specify where the directory name is                                                                       
        moveby = 40
    else:
        moveby = 36
        
    with open(args.input[0]) as ifile:
        directories = [line[:-1] for line in ifile] #don't include the '\n' character                     
    ifile.closed

    datadirs_maxLev = []
     
    if args.COM_corrected:  #Find and store all valid directories
        datadirs = [os.path.join(d,x)
                    for direc in directories
                    for d, dirs, files in os.walk(direc, topdown=True)
                    for x in files if x.endswith("rhOverM_Asymptotic_GeometricUnits_CoM.h5") and os.path.isfile(d+'/'+x)]
    
    else:
        datadirs = [os.path.join(d,x)
                    for direc in directories
                    for d, dirs, files in os.walk(direc, topdown=True)
                    for x in files if x.endswith("rhOverM_Asymptotic_GeometricUnits.h5") and os.path.isfile(d+'/'+x)]
         
    datadirs = sorted(datadirs) #sort directories
    
    for x in range(len(datadirs)-1): #Find and keep highest resolution version
        name1 = datadirs[x]
        name2 = datadirs[x+1]
        if name1[:-moveby-2]==name2[:-moveby-2] or 'BHNS' in datadirs[x]:
            pass
        else:
            datadirs_maxLev.append(name1)
    datadirs_maxLev.append(datadirs[-1])

    #Extrapolations = ['Extrapolated_N2.dir', 'Extrapolated_N3.dir', 'Extrapolated_N4.dir', 
    #                  'OutermostExtraction.dir']
    extrapolation = 'Extrapolated_N4.dir'
    
    #p = [] #Holders for linear momentum data, get 3 componenets for each time step
    pmag = [] #p magnitude
    #p_m0 = []
    #pmag_m0= []
    times_coord = [] #holders for the coordinate time values for each simulation considered
    times = [] #holders for the retarded time values, explicitly for SWSH related values
    #timebounds = []
    #simnum = [] #sxs simulation number/ simulation name
    #altnum = [] #sxs simulation designation / alternative names
    acc_com = [] #com acceleration holder
    #mag_avg = [] #How much larger on avg acc_com is than pdot/M
    #magq1 = []
    #magq1 = []
    #magqbig = []

    for datadir in datadirs_maxLev: #Each directory 
        #for extrapolation in Extrapolations:
   
        with  open(datadir[:-moveby]+'metadata.txt') as meta: #get merger time from metadata.txt
            for line in meta:
                if 'common-horizon-time' in line:
                    try:
                        t_comhor = float(line.split()[-1])
                    except ValueError:
                        pass
                if 'relaxed-measurement-time' in line:
                    try:
                        t_relaxed = float(line.split()[-1])
                    except ValueError:
                        pass
                #if 'simulation-name' in line:
                #    simnum.append(line.split()[-1][:-5])
                #if 'alternative-names' in line:
                #    altnum.append(line.split()[-1])

        with h5py.File(datadir[:-moveby]+'Horizons.h5','r') as horizons: #BH masses from apparent horizons.
            m_A = horizons['AhA.dir/ChristodoulouMass.dat'][int(t_relaxed)*2:,1] #only keep from junk radiation
            t_A = horizons['AhA.dir/ChristodoulouMass.dat'][int(t_relaxed)*2:,0] #times for m_A and m_B should be the same
            m_B = horizons['AhB.dir/ChristodoulouMass.dat'][int(t_relaxed)*2:,1]
            #m_C = horizons['AhC.dir/ChristodoulouMass.dat'][:,1]
            #t_C = horizons['AhC.dir/ChristodoulouMass.dat'][:,0]#times after common horizon found

        M = [a+b for a,b in zip(m_A,m_B)] #total mass for every timestep
        #mc = np.ndarray.tolist(m_C)
        #M = M + mc #M is now a list with all mass data

        temptime = t_A[:] - t_comhor
        #temptime2 = t_C[:] - t_comhor
        times_coord.append(temptime)#only keep times before common horizon and after junk radiation phase for COM calculated acceleration 

        #timebounds.append([2*t_relaxed, t_comhor])

        h = scri.SpEC.read_from_h5(datadir+'/'+extrapolation)
        U = h.t[:] - t_comhor #set t=0 to be at merger for all simulations
        times.append(U[(np.abs(U+t_comhor-t_relaxed).argmin()):(np.abs(U-0.0)).argmin()])#Only keep data before merger and after junk radiation phase
        if len(times[-1])>len(times_coord[-1]): #more values in retarded time array than in coordinate time array
            #need to increase size of M so can divde pdot values
            mtemp = [M[0] for idx in range(len(times[-1]) - len(times_coord[-1]))]
            M = mtemp + M
        elif len(times[-1])<len(times_coord[-1]): #more values in coordinate time than in retarded time array
            #need to decrease size of M accordingly
            M = M[len(times_coord[-1])-len(times[-1]):]
        else:#arrays are the same size, good to go
            pass

        h1 = start_with_ell_equals_zero(h)
        hdot = np.empty_like(h1)

       # try:
       #     idx_coord = np.where(temptime==np.abs(temptime-0.0).min())[0][0] #index of common horizon time for coordinate time array
       #     i = 0 #use temptime for idx_coord
       # except IndexError:
       #     try:
       #         idx_coord = np.where(temptime2==np.abs(temptime2-0.0).min())[0][0] #if common horizon time not in time array, will be the second time array for AhC
       #         i=1 #use temptime2 for idx_coord
       #     except IndexError:
       #         idx_coord = 0 #shouldn't get here, but common horizon time should be very close to 1st element of temptime2
       #         i=1

        #try:
        #    idx_retarded = np.where(U==np.abs(U-0.0).min())[0][0] #index of common horizon time for retarded time 
        #except IndexError:
        #    print(datadir[:-moveby])
        #    datadirs_maxLev.remove(datadir)
        #    continue #skip run and remove from list, badly behaved

        #if i==0:
        #    extend_prior = abs(len(U[:idx_retarded]) - len(temptime[:idx_coord])) #how many more spots to go before start time
        #    extend_post = abs(len(U[idx_retarded:]) - len(temptime[idx_coord:]) - len(temptime2)) #how many more spots to go after the end time
        #else:
        #    extend_prior = abs(len(U[:idx_retarded]) - len(temptime) - len(temptime2[:idx_coord])) #how many more spots to go before  start time                                                                                                         
        #    extend_post = abs(len(U[idx_retarded:]) - len(temptime2[idx_coord:])) #how many more sp\ots to go after the end time 

        #temptime = np.concatenate((temptime,temptime2[1:]),axis=0) #complete coordinate time array
        
        #if temptime[0]>U[0] and extend_prior!=0: #if U has more prior merger data
        #    mtemp = [M[0] for idx in range(extend_prior)]
        #    M = mtemp + M
        #else: #same size or temptime has more prior data 
        #    M = M[extend_prior:]

        #if temptime[-1]<U[-1] and extend_post!=0: #if U has more post merger data
        #    mtemp= [M[-1] for idx in range(extend_post)]
        #    M = M + mtemp
        #else: #same size or temptime has more post data
        #    M = M[:-extend_post]

        p_temp = [] #all momentum values for the current simulation
        pmag_temp = [] #all momentum magnitude values for the current simulation
       # p_m0_temp = []
       # pmag_m0_temp = []

        for j in range(h1.shape[1]):
            hdot[:,j] = (spline(U,h1[:,j].real,k=5).derivative()(U) + #real part derived
                1j*spline(U,h1[:,j].imag,k=5).derivative()(U))  #imaginary part derived, added back in 
        
        for hdot_pt in hdot:
            p_temp.append(p_multiply(hdot_pt,2,8))
            #p_m0_temp.append(p_multiply_m0(hdot_pt,2,8))
            #Each component of each element of p is returned as complex, so only need to save the real part.
            pmag_temp.append(math.sqrt(pow(abs(p_temp[-1][0]),2)+pow(abs(p_temp[-1][1]),2)+pow(abs(p_temp[-1][2]),2)))
            #pmag_m0_temp.append(math.sqrt(pow(abs(p_m0_temp[-1][0]),2)+pow(abs(p_m0_temp[-1][1]),2)+pow(abs(p_m0_temp[-1][2]),2)))

       # p.append([[item/a for item in sub ] for sub,a in zip(p_temp, M)])
        pmag.append([a/b for a,b in zip(pmag_temp,M)]) #mag. acceleration values
        #p_m0.append(p_m0_temp)
        #pmag_m0.append(pmag_m0_temp)

            #Choose color based on extrapolation
            #if 'N2' in extrapolation:
             #   col = 'g'
            #elif 'N3' in extrapolation:
             #   col = 'b'
            #elif 'N4' in extrapolation:
             #   col = 'm'
            #else:
             #   col = 'r'

        #Calculate acceleration
        
        t,com = scri.SpEC.com_motion(datadir[:-moveby]+'Horizons.h5')
        spline_x = splrep(t,com[:,0])
        spline_y = splrep(t,com[:,1])
        spline_z = splrep(t,com[:,2])

        dx2 = splev(t,spline_x,der=2)
        dy2 = splev(t,spline_y,der=2)
        dz2 = splev(t,spline_z,der=2)

        times_coord[-1] = t[:]-t_comhor
        acc_com.append([math.sqrt(x**2+y**2+z**2) for x,y,z in zip(dx2,dy2,dz2)])
#np.vstack((dx2,dy2,dz2)).T #com acceleration

        #mag_avg.append(sum(acc_com)*len(pmag)/(sum(pmag)*len(acc_com)))
#        mag_avg = sum(acc_com)*len(pmag)/(sum(pmag)*len(acc_com))
        
#        if m_A[0]/m_B[0] <= 1.05: #close to equal mass or equal mass
#            magq1.append(mag_avg)
#        else:
#            magqbig.append(mag_avg)
    colg = 'b'
    coll = 'g'

    #plt.hist(mag_avg, bins=np.logspace(np.log10(1e-5), np.log10(1e5),11), edgecolor = 'white', linewidth = 2.0)
    #plt.gca().set_xscale('log')
    #plt.title(r'$\frac{|\vec{a}_{COM}|}{|\vec{\dot{p}}|/M}$ for all runs in catalog')
    #plt.xlabel(r'Magnitude of $\frac{|\vec{a}_{COM}|}{|\vec{\dot{p}}|/M}$')
    #plt.ylabel('Frequency')
    #plt.savefig(args.filename+'_magHistogramAll.pdf', bbox_inches='tight')
    #plt.clf()

#    plt.figure(1)
#    plt.subplot(121)
#    plt.hist(magq1, bins=np.logspace(np.log10(1e-5), np.log10(1e5),11), edgecolor = 'white', linewidth = 2.0)
#    plt.gca().set_xscale('log')
#    plt.ylabel('Frequency')
#    plt.xlabel(r'Magnitude of $\frac{|\vec{a}_{COM}|}{|\vec{\dot{p}}|/M}$ for q$\leq$1.05')

#    plt.subplot(122)
#    plt.hist(magqbig, bins=np.logspace(np.log10(1e-5), np.log10(1e5),11), edgecolor = 'white', linewidth = 2.0)
#    plt.gca().set_xscale('log')
#    plt.xlabel(r'Magnitude of $\frac{|\vec{a}_{COM}|}{|\vec{\dot{p}}|/M}$ for q>1.05')
#    plt.savefig(args.filename+'_magHistogramqbig.pdf', bbox_inches='tight')


    
    for idx in range(len(datadirs_maxLev)):
        #per = [abs((item1 - item2))/abs(item1) for item1,item2 in zip(pmag_m0[idx],pmag[idx])]
        plt.semilogy(times[idx], pmag[idx],color = colg, alpha = 0.7, linewidth = 0.5 )
        plt.semilogy(times_coord[idx], acc_com[idx], color = coll, alpha = 0.7, linewidth = 0.5)
        #f.write(datadirs_maxLev[idx][42:-moveby-6]+"        "+revnum[idx]+"        "+spellnum[idx])
        #f.write("\n")
        #print("Avg m=0 contribution to |p_dot| restricted: " + 
        #      repr(sum(per[int(2*timebounds[idx][0]):int(2*timebounds[idx][1])])/len(per[int(2*timebounds[idx][0]):int(2*timebounds[idx][1])])*100)+"%")
        #print("Avg m=0 contribution to |p_dot|: "+repr(sum(per)/len(per)*100)+"%\n")
        if idx == len(datadirs_maxLev)-1:
            plt.legend([r'|$\vec{\dot{p}}$|/M', r'|$\vec{a}_{COM}$|'])
    plt.title(r'|$\vec{\dot{p}}$|/M and |$\vec{a}_{COM}$| vs time') #for '+altnum[idx])
    plt.xlabel(r'$t/M$')
    plt.ylabel(r'$Accelerations$')
    plt.savefig(args.filename+'_accmagvstime.pdf', bbox_inches='tight')#+simnum[idx]+'_accmagvstime.pdf', bbox_inches = "tight")
    plt.clf()

    return #stop here for preliminary analysis 2018/07/20


    #for idx in range(len(datadirs_maxLev)):
    #    freq = np.linspace(0.001,100000,10000000)
    #    pgram1 = signal.lombscargle(np.array(times[idx]),np.array(pmag_m0[idx]),freq)
    #    pgram2 = signal.lombscargle(np.array(times[idx]),np.array(pmag[idx]),freq)
    #    plt.plot(freq,pgram1,color = "b")
    #    plt.plot(freq,pgram2,color = "g")
    #plt.title(r'Periodogram for $|\vec{\dot{p}}|$ vs time')
    #plt.xlabel(r'Freq')
    #plt.legend(['With m=0','Without m=0'])
    #plt.ylabel(r'$|\vec{\dot{p}_m0}|/|\vec{\dot{p}}|$')
    #plt.savefig(args.filename+'_pmagperiodogram.pdf', bbox_inches = "tight")
    #plt.clf()

    for idx in range(len(datadirs_maxLev)):
        px= [item[0] for item in p[idx]]
        px_pos = []
        px_neg = []
        for idy in range(len(px)):
            if px[idy]>1e-16:
                px_pos.append(px[idy])
                px_neg.append(1e-16)
            else:
                px_pos.append(1e-16)
                px_neg.append(-px[idy])
        plt.semilogy(times[idx], px_pos, color = colg, alpha = 0.5, linewidth = 0.5)
        plt.semilogy(times[idx], px_neg, color = coll, alpha = 0.5, linewidth = 0.5)
        #perx = [abs(item1[0]-item2[0])/abs(item1[0]) for item1,item2 in zip(p_m0[idx],p[idx])]
        #plt.semilogy(times[idx],perx,color = colg, alpha = 0.5, linewidth = 0.5)
        #print("Avg m=0 contribution to p_dot_x: "+repr(sum(perx)/len(perx)*100)+"%\n")
        plt.title(r'$\dot{p}_x$ vs time for '+altnum[idx])
        plt.xlabel(r'$t/M$')
        plt.ylabel(r'$\dot{p}_x$')
        plt.savefig(args.filename+simnum[idx]+'_pxvstime.pdf', bbox_inches="tight")
        plt.clf()
    
    for idx in range(len(datadirs_maxLev)):
        py= [item[1] for item in p[idx]]
        py_pos = []
        py_neg = []
        for x in py:
            if x>1e-16:
                py_pos.append(x)
                py_neg.append(1e-16)
            else:
                py_pos.append(1e-16)
                py_neg.append(-x)

        plt.semilogy(times[idx], py_pos, color = colg, alpha = 0.5, linewidth = 0.5)
        plt.semilogy(times[idx], py_neg, color = coll, alpha = 0.5, linewidth = 0.5)
        #pery = [abs(item1[1]-item2[1])/abs(item1[1]) for item1,item2 in zip(p_m0[idx],p[idx])]
        #plt.semilogy(times[idx],pery,color = colg, alpha = 0.5, linewidth = 0.5)
        #print("Avg m=0 contribution to p_dot_y: "+repr(sum(pery)/len(pery)*100)+"%\n")
        plt.title(r'$\dot{p}_y$ vs time for '+altnum[idx])
        plt.xlabel(r'$t/M$')
        plt.ylabel(r'$\dot{p}_y$')
        plt.savefig(args.filename+simnum[idx]+'_pyvstime.pdf', bbox_inches="tight")
        plt.clf()

    for idx in range(len(datadirs_maxLev)):
        pz= [item[2] for item in p[idx]]
        pz_pos = []
        pz_neg = []
        for x in pz:
            if x>1e-16:
                pz_pos.append(x)
                pz_neg.append(1e-16)
            else:
                pz_pos.append(1e-16)
                pz_neg.append(-x)

        plt.semilogy(times[idx], pz_pos, color = colg, alpha = 0.5, linewidth = 0.5)
        plt.semilogy(times[idx], pz_neg, color = coll, alpha = 0.5, linewidth = 0.5)
        #perz = [abs(item1[2]-item2[2])/abs(item1[2]) for item1,item2 in zip(p_m0[idx],p[idx])]
        #plt.semilogy(times[idx],perz,color = colg, alpha = 0.5, linewidth = 0.5)
        #print("Avg m=0 contribution to p_dot_z: "+repr(sum(perz)/len(perz)*100)+"%\n")
        plt.title(r'$\dot{p}_z$ vs time for '+altnum[idx])
        plt.xlabel(r'$t/M$')
        plt.ylabel(r'$\dot{p}_z$')
        plt.savefig(args.filename+simnum[idx]+'_pzvstime.pdf', bbox_inches="tight")
        plt.clf()

                 
if __name__=="__main__":
    main()
