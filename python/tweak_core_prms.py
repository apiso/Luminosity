from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk, params, yr
from utils.parameters import FT, FSigma, mstar, Y, delad, rhoc, Mc, rc, \
     gamma, Y, a
import numpy as np
from numpy import pi
import sys
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
from scipy import integrate, interpolate, optimize
from scipy.integrate import odeint
from types import FunctionType as function
from scipy.interpolate import interp1d
from scipy.optimize import brentq, root
from profiles_SG import atmload, prms
from profiles_no_SG import atmload as atmloadnoSG
from cooling import cooling_global
from luminosity_numerical_SG import shoot
from luminosity_numerical_no_SG import shoot as shootnoSG
import evaporation_4 as evap4
reload(evap4)
import utils.constants as c

#prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
#              Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
#              Td = Tdisk(a, FT), kappa = kdust) #gas, disk and core parameters
                        #for specific values imported from parameters.py 
                        


def delradfn(p, m, T, L, prms): #radiative temperature gradient
    rho = p / (prms.R * T)
    return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)

def Del(p, m, T, L, prms): #del = min(delad, delrad)
    return min(prms.delad, delradfn(p, m, T, L, prms))
    
    
td = 3e6
n = 500
tol = 1e-24
SG = 1
filename = 'a1Mc10'

model, param, prof = atmload(filename, prms)

L, t, Mf, rf, MBd, r, P, T, rho, m, delrad, Eg, U, Mcb, MB, rcb, RB, Pc, Pcb, Tc, Tcb, \
        Egcb, Ucb, Etotcb, EgB, UB, EtotB, Etot, Ecool, Eevap, indf, flag =  evap4.mass_loss_2(filename, prms, td, n, tol, SG)
        
E0 = 0#G * prms.Mco**2 / prms.rco
Mf = 10.120477756628246        
        
rfit = c.RHill(Mf * Me, prms.a) #np.array(c.RHill(Mf * Me, prms.a), c.RB(Mf * Me, prms.a)).min()
         
R = np.logspace(np.log10(prms.rco), np.log10(rfit), n)
        
def f(x, r):
    #    
    #"""
    #structure eqns. "x" = [p , T , m, L, Eg, U, Iu],
    #    with Iu the 3p/rho dm integral in the virial theorem
    #dp/dr = - G * m * P / (r**2 * R * T),
    #dT/dr = - del * G * m / (R * r**2)
    ###dm/dr = 4 * pi * r**2 * p / (R * T)
    #dL/dr = 0
    ###dEg/dr = - 4 * pi * G * m * r * P / (R * T)
    #dU/dr = 4 * pi * r**2 * P * Cv / R
    #"""
        
    return np.array([ - G * x[2] * x[0] / (r**2 * prms.R * x[1]), \
                             - Del(x[0], x[2], x[1], x[3], prms) * G * x[2] / \
                             (prms.R * r**2),
                             4 * pi * r**2 * x[0] / (prms.R * x[1]), \
                             0, \
                             - 4 * pi * G * x[2] * r * x[0] / (prms.R * x[1]), \
                             4 * pi * r**2 * x[0] * prms.Cv / prms.R]) 

        #R = np.logspace(np.log10(prms.rco), np.log10(rcb * Re), n)
        
        #time = t
        

        #dt = - Eevap / L
        #time = time + dt
    
def trial(Ptry, Ttry, Ltry, filename = filename, prms = prms, td = td, n = n, tol = tol, SG = SG):
    

    
    y = odeint(f, [Ptry, Ttry, prms.Mco, Ltry, -E0, E0], R)
    Mtot = y[:,2][-1]
        
        #deltaM = 4 / np.pi * np.arctan(Mtot / m[-2]) - 1
        #deltaM = 4 / np.pi * np.arctan(Mtot / m[indf]) - 1
        #relative error; use of the arctan ensures deltaL stays between -1 and 1
        #if math.isnan(deltaM): #used to get rid of possible divergences
        #        deltaM = 1.
        
        #return deltaM 





#        
#        
#    tcoreguess = 0.9 * Tc
#    Pcoreguess = Pc/ 10   
#    Lguess = L*1e-1
#
#    #Tmatch = root(delta, [tcoreguess, Pcoreguess, Lguess])
        #Lmatch = root(delta, L1, method = 'hybr')
        #Ecoolnew = y[1][4] + y[1][5]
        
    rnew = R
    Pnew = y[:,0]
    Tnew = y[:,1]
    mnew = y[:,2]
    Egnew = y[:,4]
    Unew = y[:,5]
    #Lnew = Lmatch.x

    delradnew = 0 * np.ndarray(shape = len(Pnew), dtype = float)
    for i in range(len(delradnew)):
        delradnew[i] = delradfn(Pnew[i], mnew[i], Tnew[i], Ltry, prms)

    rhonew = Pnew / (prms.R * Tnew)

    #interpolation functions to find the RCB
    fr = interp1d(delradnew[::-1], rnew[::-1])
    fP = interp1d(delradnew[::-1], Pnew[::-1])
    fT = interp1d(delradnew[::-1], Tnew[::-1])
    fm = interp1d(delradnew[::-1], mnew[::-1])
    fEg = interp1d(delradnew[::-1], Egnew[::-1])
    fU = interp1d(delradnew[::-1], Unew[::-1])

    rcbnew = float(fr(prms.delad))
    Pcbnew = float(fP(prms.delad))
    Tcbnew = float(fT(prms.delad))
    Mcbnew = float(fm(prms.delad))
    Egcbnew = float(fEg(prms.delad))
    Ucbnew = float(fU(prms.delad))
    Etotcbnew = Egcbnew + Ucbnew
    
    return rcbnew / Re, Mcbnew / Me, Etotcbnew
    
    
Ptrial = np.logspace(np.log10(Pc*1e-3), np.log10(Pc*1e3), 50)
Ttrial = np.linspace(Tc / 1.1, Tc * 1.1, 50)
Ltrial = np.logspace(np.log10(L*1e-3), np.log10(L*1e3), 50)

fp = open('../dat/SG/k_dust/grid_a1Mc10.txt', 'wb')
fp.write("rcb      Mcb      Etotcb \n")
for i in range(50):
    for j in range(50):
        for k in range(50):
            try:
                temp = trial(Ptrial[i], Ttrial[j], Ltrial[k])
                fp.write(" %.3f " % temp[0])
                fp.write(" %.3f " % temp[1])
                fp.write(" %.3f \n" % temp[2])
            except ValueError:
                pass
            
            print i, j, k
            
fp.close()







