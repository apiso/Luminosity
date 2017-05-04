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
from profiles_SG import atmload
from profiles_no_SG import atmload as atmloadnoSG
from cooling import cooling_global
from luminosity_numerical_SG import shoot
from luminosity_numerical_no_SG import shoot as shootnoSG


#prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
#              Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
#              Td = Tdisk(a, FT), kappa = kdust) #gas, disk and core parameters
                        #for specific values imported from parameters.py 

def delradfn(p, m, T, L, prms): #radiative temperature gradient
    rho = p / (prms.R * T)
    return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)

def Del(p, m, T, L, prms): #del = min(delad, delrad)
    return min(prms.delad, delradfn(p, m, T, L, prms))

def mass_loss(filename, prms, td = 3e6):
    
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
    
    
    for i in range(len(param)):
        if param.MB[i] <= MBd and param.MB[i + 1] > MBd:
            break
    ind = i
    
    for i in range(0, ind)[::-1]:
        Eevap = np.abs(param.EtotB[ind] - param.EtotB[i])
        Ecool = np.abs(param.EtotB[i])
        if Ecool - Eevap <= 0:
            break
    indf = i
            
    len1 = len(param[:ind+1])
    len2 = len(param[indf:ind][::-1])
    nMpoints = len1 + len2
    n = len(prof[0])
        
    
    model2 = np.array([(prms.Mco, prms.rco, prms.a, prms.delad, prms.Y, \
                          prms.gamma, prms.R, prms.Cv, prms.Pd, prms.Td, \
                          prms.kappa)], \
                        dtype = [('Mco', float), ('rco', float), ('a', float), \
                                 ('delad', float), ('Y', float), \
                                 ('gamma', float), \
                                 ('R', float), ('Cv', float), ('Pd', float), \
                                 ('Td', float), ('kappa', function)])
    
    prof2 = np.recarray(shape = (nMpoints, n), \
                          dtype = [('r', float), ('P', float), ('t', float), \
                                   ('m', float), ('rho', float), ('delrad', float), \
                                   ('Eg', float), ('U', float)])
    param2 = np.recarray(\
        shape = (nMpoints), \
        dtype = [('Mtot', float), ('Mcb', float), ('MB', float), ('rcb', float),\
                 ('RB', float), ('RHill', float), ('Pc', float), ('Pcb', float),\
                 ('PB', float), ('Tc', float), ('Tcb', float), ('TB', float), \
                 ('Egcb', float), ('Ucb', float), ('Etotcb', float), \
                 ('EgB', float), ('UB', float), ('EtotB', float), \
                 ('EgHill', float), ('UHill', float), ('EtotHill', float), \
                 ('L', float), ('vircb', float), ('virHill', float), \
                 ('err', float)])
                 
    for i in range(len1):
        param2[i] = param[i]
        prof2[i] = prof[i]
    for i in range(len2):
        param2[len1 + i] = param[indf:ind][::-1][i]
        prof2[len1 + i] = prof[indf:ind][::-1][i]
        
    dt2 = np.append(dt[:ind+1], dt[indf:ind][::-1])
    time2 = []
    for i in range(len(dt2)):
        time2 = np.append(time2, sum(dt2[:i + 1]))
    
    return model2, param2, prof2, time2
    
    
    
def mass_loss_2(filename, prms, td = 3e6, n = 500, tol = 1e-24, SG = 1):
    
    """
    
    Determines the mass loss due to spontaneous mass loss after disk dispersal 
    as the disk density goes to zero. Calculates the timescale on which this 
    process happens, the atmospheric mass that is evaporated, the change in
    atmospheric luminosity, radius, pressure etc.
    
    """
    if SG == 1:
        model, param, prof = atmload(filename, prms)
    else:
        model, param, prof = atmloadnoSG(filename, prms)
        param.MB = param.Mtot
    dt = cooling_global(param, prof, model, out='rcb')[1]
    
    time = []
    
    for i in range(len(dt)):
        time = np.append(time, sum(dt[:i + 1]))
    if SG == 1:    
        f = interp1d(time / yr, param.MB[:-1])
        MBd = float(f(td))
    else:
        f = interp1d(time / yr, param.Tc[:-1])
        Tcd = float(f(td))
        fMB = interp1d(param.Tc, param.MB)
        MBd = float(fMB(Tcd))    
    
#    fMrcb = interp1d(param.MB, param.Mcb)
#    Mrcbd = float(fMrcb(MBd))
#    
#    frcb = interp1d(param.MB, param.rcb)
#    rcbd = float(frcb(MBd))
#
#    fRB = interp1d(param.MB, param.RB)
#    RBd = float(fRB(MBd))
#   
    if SG == 1: 
        fL = interp1d(param.MB, param.L)
        Ld = float(fL(MBd)) 
    else:
        fL = interp1d(param.Tc, param.L)
        Ld = float(fL(Tcd))    
    if SG == 1:
        sol = shoot(MBd * Me, Ld*1e-1, Ld*1e1, n, tol, prms)
        r, P, T, m, rho, delrad, Eg, U, Mi, Mcb, MB, rcb, \
            RB, rfit, Pc, Pcb, PB, Tc, Tcb, TB, Egcb, Ucb, Etotcb, \
            EgB, UB, EtotB, EgHill, UHill, EtotHill, L, vircb, virHill, err = sol
        Etot = Eg + U
    
    else:
        sol = shootnoSG(Tcd, Ld*1e-1, Ld*1e1, n, tol, prms)
        r, P, T, rho, m, delrad, Eg, U, Mcb, MB, rcb, RB, Pc, Pcb, Tc, Tcb, \
            Egcb, Ucb, Etotcb, EgB, UB, EtotB, L, err = sol
        Etot = Eg + U
    
    for i in range(n)[::-1]:
        Eevap = np.abs(EtotB - Etot[i])
        Ecool = np.abs(Etot[i])
                
        if Ecool - Eevap < 0:
            break
    indf = i
    
    Mf = m[indf] / Me
    rf = r[indf] / model.rco[0]
    t = -(EtotB - Etot[indf])/L
    
    return L, t, Mf, rf, MBd
    
    
    
#    
#    
#    def f(x, r):
#    #    
#    #"""
#    #structure eqns. "x" = [p , T , m, L, Eg, U, Iu],
#    #    with Iu the 3p/rho dm integral in the virial theorem
#    #dp/dr = - G * m * P / (r**2 * R * T),
#    #dT/dr = - del * G * m / (R * r**2)
#    ###dm/dr = 4 * pi * r**2 * p / (R * T)
#    #dL/dr = 0
#    ###dEg/dr = - 4 * pi * G * m * r * P / (R * T)
#    #dU/dr = 4 * pi * r**2 * P * Cv / R
#    #"""
#        
#        return np.array([ - G * x[2] * x[0] / (r**2 * prms.R * x[1]), \
#                             - Del(x[0], x[2], x[1], x[3], prms) * G * x[2] / \
#                             (prms.R * r**2),
#                             4 * pi * r**2 * x[0] / (prms.R * x[1]), \
#                             0, \
#                             - 4 * pi * G * x[2] * r * x[0] / (prms.R * x[1]), \
#                             4 * pi * r**2 * x[0] * prms.Cv / prms.R]) 
#    #E0 = G * Mi**2 / rfit
#    R = np.logspace(np.log10(RBd*Re), np.log10(model.rco), nMpoints)
#        #radius grid
#    r = R[:2]    
#    
#    y = odeint(f, [prms.Pd, prms.Td, MBd*Me, Ld, EgBd, UBd], r)
#    
#    Mi = y[1][2]
#    Egabs = -(y[0][4] - y[1][4])
#    Etoti = y[1][4]
#
#    
#    sol = shoot(Mi, Ld*1e-3, Ld*1e3, n, tol, prms)
#    
#    i = 0
#    
#    param2.Mtot[i], param2.Mcb[i], param2.MB[i], param2.rcb[i], param2.RB[i], \
#                       param2.RHill[i], param2.Pc[i], param2.Pcb[i], param2.PB[i],\
#                       param2.Tc[i], param2.Tcb[i], param2.TB[i], param2.Egcb[i], \
#                       param2.Ucb[i], param2.Etotcb[i], param2.EgB[i], \
#                       param2.UB[i], param2.EtotB[i], param2.EgHill[i], \
#                       param2.UHill[i], param2.EtotHill[i], param2.L[i], \
#                       param2.vircb[i], param2.virHill[i], param2.err[i] = sol[8:]
#
#    for k in range(n):
#        prof2.r[i, k], prof2.P[i, k], prof2.t[i, k], prof2.m[i, k], \
#                      prof2.rho[i, k], prof2.delrad[i, k], \
#                      prof2.Eg[i, k], prof2.U[i, k] = \
#                      sol[0][k], sol[1][k], sol[2][k], sol[3][k], sol[4][k], \
#                      sol[5][k], sol[6][k], sol[7][k]  
#                      
#    Mi = param2.MB[i] * Me
#    Li = param2.L[i]
#    Etoti = np.abs(param2.EtotB[i])
#    i = 1
#    
#    while(np.abs(Etoti) - Egabs) >= 0:
#    
#        r = R[i:i+2]    
#    
#        y = odeint(f, [prms.Pd, prms.Td, Mi, Li, param2.EgB[i-1], param2.UB[i-1]], r)
#    
#        Mi = y[1][2]
#        Li = param2.L[i - 1]
#        Egabs = Egabs -(y[0][4] - y[1][4])
#        Etoti = np.abs(param2.EtotB[i-1])
#    
#        sol = shoot(Mi, Li*1e-2, Li*1e2, n, tol, prms)
#    
#    
#        param2.Mtot[i], param2.Mcb[i], param2.MB[i], param2.rcb[i], param2.RB[i], \
#                       param2.RHill[i], param2.Pc[i], param2.Pcb[i], param2.PB[   i],\
#                       param2.Tc[i], param2.Tcb[i], param2.TB[i], param2.Egcb[i], \
#                       param2.Ucb[i], param2.Etotcb[i], param2.EgB[i], \
#                       param2.UB[i], param2.EtotB[i], param2.EgHill[i], \
#                       param2.UHill[i], param2.EtotHill[i], param2.L[i], \
#                       param2.vircb[i], param2.virHill[i], param2.err[i] = sol[8:]
#
#        for k in range(n):
#            prof2.r[i, k], prof2.P[i, k], prof2.t[i, k], prof2.m[i, k], \
#                      prof2.rho[i, k], prof2.delrad[i, k], \
#                      prof2.Eg[i, k], prof2.U[i, k] = \
#                      sol[0][k], sol[1][k], sol[2][k], sol[3][k], sol[4][k], \
#                      sol[5][k], sol[6][k], sol[7][k]  
#                      
#        i += 1
#        
#        print i
#    
#    
#    
#    
#    #return MBd, Ld, EtotBd, RBd, y
    
    
    
    
    
    
    