from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk, params, yr
from utils.parameters import FT, FSigma, mstar, Y, delad, rhoc, Mc, rc, \
     gamma, Y, a
import numpy as np
from numpy import pi
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
from scipy import integrate, interpolate, optimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from profiles_SG import atmload
from cooling import cooling_global


prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
              Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
              Td = Tdisk(a, FT), kappa = kdust) #gas, disk and core parameters
                        #for specific values imported from parameters.py 

def delradfn(p, m, T, L, prms = prms): #radiative temperature gradient
    rho = p / (prms.R * T)
    return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)

def Del(p, m, T, L, prms = prms): #del = min(delad, delrad)
    return min(prms.delad, delradfn(p, m, T, L, prms))

def mass_loss(filename, prms = prms, td = 3e6):
    
    """
    
    Determines the mass loss due to spontaneous mass loss after disk dispersal 
    as the disk density goes to zero. Calculates the timescale on which this 
    process happens, the atmospheric mass that is evaporated, the change in
    atmospheric luminosity, radius, pressure etc.
    
    """
    
    model, param, prof = atmload(filename, prms)
    dt = cooling_global(param, prof, model, out='rcb')[1]
    
    time = []
    
    for i in range(len(dt)):
        time = np.append(time, sum(dt[:i + 1]))
        
    f = interp1d(time / yr, param.MB[:-1])
    MBd = float(f(td))
    
    fMrcb = interp1d(param.MB, param.Mcb)
    Mrcbd = float(fMrcb(MBd))
    
    frcb = interp1d(param.MB, param.rcb)
    rcbd = float(frcb(MBd))

    fRB = interp1d(param.MB, param.RB)
    RBd = float(fRB(MBd))
    
    fL = interp1d(param.MB, param.L)
    Ld = float(fL(MBd))
    
    fEgB = interp1d(param.MB, param.EgB)
    EgBd = float(fEgB(MBd))
    
    fUB = interp1d(param.MB, param.UB)
    UBd = float(fUB(MBd))
    
    fEtotB = interp1d(param.MB, param.EtotB)
    EtotBd = float(fEtotB(MBd))
    
    
    
    def f(x, r):
    #    
    #"""
    #structure eqns. "x" = [p , T , m, L, Eg, U, Iu],
    #    with Iu the 3p/rho dm integral in the virial theorem
    #dp/dr = - G * m * P / (r**2 * R * T),
    #dT/dr = - del * G * m / (R * r**2)
    #dm/dr = 4 * pi * r**2 * p / (R * T)
    #dL/dr = 0
    #dEg/dr = - 4 * pi * G * m * r * P / (R * T)
    #dU/dr = 4 * pi * r**2 * P * Cv / R
    #dIu/dr = 12 * pi * P * r**2
    #"""
        
        return np.array([ - G * x[2] * x[0] / (r**2 * prms.R * x[1]), \
                             - Del(x[0], x[2], x[1], x[3], prms) * G * x[2] / \
                             (prms.R * r**2),
                             4 * pi * r**2 * x[0] / (prms.R * x[1]), \
                             0, \
                             - 4 * pi * G * x[2] * r * x[0] / (prms.R * x[1]), \
                             4 * pi * r**2 * x[0] * prms.Cv / prms.R]) 
    #E0 = G * Mi**2 / rfit
    r = np.logspace(np.log10(RBd*Re), np.log10(0.9*RBd*Re), 2)
        #radius grid
    y = odeint(f, [prms.Pd, prms.Td, MBd * Me, Ld, 0, 0], r)
    
    return MBd, Ld, EtotBd, RBd, y
    
    
    
    
    
    
    