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
from types import FunctionType as function
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from profiles_no_SG import atmload
from cooling import cooling_global
from luminosity_numerical_no_SG import shoot


prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
              Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
              Td = 1e3, kappa = kdust) #gas, disk and core parameters
                        #for specific values imported from parameters.py 

def delradfn(p, m, T, L, prms = prms): #radiative temperature gradient
    rho = p / (prms.R * T)
    return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)

def Del(p, m, T, L, prms = prms): #del = min(delad, delrad)
    return min(prms.delad, delradfn(p, m, T, L, prms))

def mass_loss(filename, prms = prms, td = 3e6, tol = 1e-24, n = 100, nTcpoints = 500):
    
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
    
    param.MB = param.Mtot
    param.RB = param.rout
    param.EtotB = param.Etotout  
    param.EgB = param.Egout
    param.UB = param.Uout  
                
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
    
    
    model2 = np.array([(prms.Mco, prms.rco, prms.a, prms.delad, prms.Y, \
                          prms.gamma, prms.R, prms.Cv, prms.Pd, prms.Td, \
                          prms.kappa)], \
                        dtype = [('Mco', float), ('rco', float), ('a', float), \
                                 ('delad', float), ('Y', float), \
                                 ('gamma', float), \
                                 ('R', float), ('Cv', float), ('Pd', float), \
                                 ('Td', float), ('kappa', function)])
    
    prof2 = np.recarray(shape = (nTcpoints, n), \
                          dtype = [('r', float), ('P', float), ('t', float), \
                                   ('rho', float), ('m', float), ('delrad', float), \
                                   ('Eg', float), ('U', float)])
    param2= np.recarray(\
        shape = (nTcpoints), \
        dtype = [('Mcb', float), ('Mtot', float), ('rcb', float),\
                 ('rout', float), ('Pc', float), ('Pcb', float),\
                 ('Tc', float), ('Tcb', float), \
                 ('Egcb', float), ('Ucb', float), ('Etotcb', float), \
                 ('Egout', float), ('Uout', float), ('Etotout', float), \
                 ('L', float), ('err', float)])
    
    
    
    
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
        
        return np.array([ - G * prms.Mco * x[0] / (r**2 * prms.R * x[1]), \
                            - Del(x[0], prms.Mco, x[1], x[2], prms) * G * prms.Mco / \
                            (prms.R * r**2),
                            #4 * pi * r**2 * x[0] / (prms.R * x[1]), \
                            0, \
                            - 4 * pi * G * prms.Mco * r * x[0] / (prms.R * x[1]), \
                            4 * pi * r**2 * x[0] * prms.Cv / prms.R]) 
    #E0 = G * Mi**2 / rfit
    R = np.logspace(np.log10(RBd*Re), np.log10(0.9*RBd*Re), 2)
        #radius grid
    r = R#[:2]    
    
    y = odeint(f, [prms.Pd, prms.Td, Ld, 0, 0], r)
    
    Ti = y[1][1]
    Egabs = y[1][3]

    prms2 = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
              Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
              Td = Ti, kappa = kdust)    
            
    sol = shoot(Ti, Ld*1e-3, Ld*1e3, n, tol, prms2)
    
    i = 0
    
    param2.Mtot[i], param2.Mcb[i], param2.MB[i], param2.rcb[i], param2.RB[i], \
                       param2.RHill[i], param2.Pc[i], param2.Pcb[i], param2.PB[i],\
                       param2.Tc[i], param2.Tcb[i], param2.TB[i], param2.Egcb[i], \
                       param2.Ucb[i], param2.Etotcb[i], param2.EgB[i], \
                       param2.UB[i], param2.EtotB[i], param2.EgHill[i], \
                       param2.UHill[i], param2.EtotHill[i], param2.L[i], \
                       param2.vircb[i], param2.virHill[i], param2.err[i] = sol[8:]

    for k in range(n):
        prof2.r[i, k], prof2.P[i, k], prof2.t[i, k], prof2.m[i, k], \
                      prof2.rho[i, k], prof2.delrad[i, k], \
                      prof2.Eg[i, k], prof2.U[i, k] = \
                      sol[0][k], sol[1][k], sol[2][k], sol[3][k], sol[4][k], \
                      sol[5][k], sol[6][k], sol[7][k]  
                      
    Mi = param2.MB[i] * Me
    Li = param2.L[i]
    Etoti = np.abs(param2.EtotB[i])
    i = 1
    
    while(np.abs(Etoti) - Egabs) >= 0:
    
        r = R[i:i+2]    
    
        y = odeint(f, [prms.Pd, prms.Td, Mi, Li, 0, 0], r)
    
        Mi = y[1][2]
        Li = param2.L[i - 1]
        Egabs = Egabs + y[1][4]
        Etoti = np.abs(param2.EtotB[i-1])
    
        sol = shoot(Mi, Li*1e-2, Li*1e2, n, tol, prms)
    
    
        param2.Mtot[i], param2.Mcb[i], param2.MB[i], param2.rcb[i], param2.RB[i], \
                       param2.RHill[i], param2.Pc[i], param2.Pcb[i], param2.PB[   i],\
                       param2.Tc[i], param2.Tcb[i], param2.TB[i], param2.Egcb[i], \
                       param2.Ucb[i], param2.Etotcb[i], param2.EgB[i], \
                       param2.UB[i], param2.EtotB[i], param2.EgHill[i], \
                       param2.UHill[i], param2.EtotHill[i], param2.L[i], \
                       param2.vircb[i], param2.virHill[i], param2.err[i] = sol[8:]

        for k in range(n):
            prof2.r[i, k], prof2.P[i, k], prof2.t[i, k], prof2.m[i, k], \
                      prof2.rho[i, k], prof2.delrad[i, k], \
                      prof2.Eg[i, k], prof2.U[i, k] = \
                      sol[0][k], sol[1][k], sol[2][k], sol[3][k], sol[4][k], \
                      sol[5][k], sol[6][k], sol[7][k]  
                      
        i += 1
        
        print i
    
    
    
    
    #return MBd, Ld, EtotBd, RBd, y
    
    
    
    
    
    
    