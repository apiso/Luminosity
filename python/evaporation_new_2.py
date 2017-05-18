from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk, params, yr
from utils.parameters import FT, FSigma, mstar, Y, delad, rhoc, Mc, rc, \
     gamma, Y, a
import numpy as np
import sys
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
from scipy.optimize import brentq, root, fsolve
from profiles_SG import atmload
from cooling import cooling_global
from luminosity_numerical_SG import shoot
from utils import constants as c


#prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
#              Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
#              Td = Tdisk(a, FT), kappa = kdust) #gas, disk and core parameters
                        #for specific values imported from parameters.py 

def delradfn(p, m, T, L, prms): #radiative temperature gradient
    rho = p / (prms.R * T)
    return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)

def Del(p, m, T, L, prms): #del = min(delad, delrad)
    return min(prms.delad, delradfn(p, m, T, L, prms))

def mass_loss(filename, prms, td = 3e6, tol = 1e-24, n = 500, nMpoints = 5000):
    
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
    #E0 = G * Mi**2 / rfit
    R = np.logspace(np.log10(RBd*Re), np.log10(model.rco), n)
        #radius grid
    r = R#[:2]    
    
    #y = odeint(f, [prms.Pd, prms.Td, MBd*Me, Ld, EgBd, UBd], r)
    
    sol = shoot(MBd * Me, Ld*1e-3, Ld*1e3, n, tol, prms)
    
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
                      

    
        
    i = 0
    Eevap = 0
    Ecool = param2.EtotB[0]
    dt2 = 0
    time = [1e-3]
    time2 = [0]
    flag = 0
    #mass = np.linspace(model.Mco, MBd * Me, n)
    
    while (time2[-1] <= time[-1]) or time[-1] < 0:
    #while(np.abs(Ecool) - np.abs(Eevap)) >= 0 and flag == 0:
        
        mass = np.linspace(model.Mco, param2.MB[i] * Me, n)
        RBondi = c.RB(mass[-2], model.a)
        Rout = c.RHill(mass[-2], model.a)
        rfit = min(RBondi, Rout)
        rnew = np.logspace(np.log10(model.rco), np.log10(rfit), n)
        
    
        Pc, Tc, L = param2.Pc[i], param2.Tc[i], param2.L[i]
        
        def delta(x):  
        
        #Pcore = x[0]   
        #tcore = x[1]
        #lum = x[2]   
           
            ynew = odeint(f, [x[0], x[1], model.Mco, x[2], 0, 0], rnew)  
        #y1 = odeint(f, [Pcguess, Tguess, prms.Mco, lum, -E0, E0], [prms.rco, r[indf]])                        
        
            Pressure = ynew[:,0]
            Temp = ynew[:,1]
            Mass = ynew[:,2]
            lum = ynew[:,3]
        
            delradnew = 0 * np.ndarray(shape = len(rnew), dtype = float)
            for j in range(len(delradnew)):
                delradnew[j] = delradfn(Pressure[j], Mass[j], Temp[j], lum[j], prms)
                
            fT = interp1d(delradnew[::-1], Temp[::-1])
            Tcbnew = float(fT(prms.delad))
                
            deltaT = 4 / np.pi * np.arctan(Mass[-1] / mass[-2]) - 1
            deltaM = 4 / np.pi * np.arctan(Temp[-1] / model.Td[0]) - 1
            deltaL = 4 / np.pi * np.arctan(Tcbnew / (1.5275 * model.Td[0])) - 1
        
        #deltaM = 4 / np.pi * np.arctan(Mtot / m[indf]) - 1
        #relative error; use of the arctan ensures deltaL stays between -1 and 1
        #if math.isnan(deltaM): #used to get rid of possible divergences
        #        deltaM = 1.
        
            err = (deltaT, deltaM, deltaL)
            return err
    
        Pctry, Tctry, Ltry = param2.Pc[i], param2.Tc[i], param2.L[i]
        x0 = (Pctry, Tctry, Ltry)  
        match = root(delta, x0)
        it = 0
        while (match.success) == False:
            Pctry, Tctry, Lctry = Pctry / 1.0001, Tctry / 1.0001, Ltry / 1.0001
            x0 = (Pctry, Tctry, Ltry)  
            match = root(delta, x0)
            it += 1
            if it == 100:
                break
        #if match.success == False:
        #    flag = 1
            #print "Nope! Try different initial guesses."
            #sys.exit()
        #else:  
        Pcmatch, Tcmatch, Lmatch =  match.x
    
        ynew = odeint(f, [Pcmatch, Tcmatch, model.Mco, Lmatch, 0, 0], rnew)
        Ecool = ynew[:,4][-1] + ynew[:,5][-1]
        Eevap = param2.EtotB[0] - Ecool
        dt = - Eevap / Lmatch
        time = np.append(time,  dt)

        Pnew = ynew[:,0]
        Tnew = ynew[:,1]
        mnew = ynew[:,2]
        Egnew = ynew[:,4]
        Unew = ynew[:,5]
        
             


        delradnew = 0 * np.ndarray(shape = len(Pnew), dtype = float)
        for j in range(len(delradnew)):
            delradnew[j] = delradfn(Pnew[j], mnew[j], Tnew[j], Lmatch, prms)

        rhonew = Pnew / (prms.R * Tnew)
        
        dt2 = dt2 + (param2.MB[i]*Me - mnew[-1]) / (4 * np.pi * rhonew[-1] * rfit**2 * \
                np.sqrt(model.R * model.Td))
        time2 = np.append(time2, dt2)

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

        EgHillnew = Egnew[-1]
        UHillnew = Unew[-1]
        EtotHillnew = EgHillnew + UHillnew

        Pcnew = Pnew[0]
        Tcnew = Tnew[0]

        dRBondi = rnew - G * mnew / (prms.R * prms.Td)
        #r - G m(r)/(R delad) = 0 at RB, so dRBondi(r) = 0 gives RB
    
        if dRBondi[-1] > 0: #ensures RB < RHill
            fRBondi = interp1d(dRBondi, mnew)
            MBnew = fRBondi(0)
            RBnew = (G * MBnew) / (prms.R * prms.Td)
            fPB = interp1d(mnew, Pnew)
            fTB = interp1d(mnew, Tnew)
            fEgB = interp1d(mnew, Egnew)
            fUB = interp1d(mnew, Unew)
    
            PBnew = float(fPB(MBnew))
            TBnew = float(fTB(MBnew))
            EgBnew = float(fEgB(MBnew))
            UBnew = float(fUB(MBnew))
            EtotBnew = EgBnew + UBnew
        else: #if RB > RHill, we are outside our boundaries, so set all Bondi
            #values to the Hill values 
            MBnew, RBnew, PBnew, TBnew, EgBnew, UBnew, EtotBnew = \
                mnew[-1], rnew[-1], Pnew[-1], Tnew[-1], EgHillnew, UHillnew, EtotHillnew
                
        sol = rnew, Pnew, Tnew, mnew, rhonew, delradnew, Egnew, Unew, mnew[-1] / Me, \
            Mcbnew / Me, MBnew / Me, rcbnew / Re, RBnew / Re, rfit / Re, \
                Pcnew, Pcbnew, PBnew, Tcnew, Tcbnew, TBnew, Egcbnew, Ucbnew, Etotcbnew, \
                    EgBnew, UBnew, EtotBnew, EgHillnew, UHillnew, EtotHillnew, Lmatch, 0, 0, 0
        i += 1   
        
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
        print i
    
    paramfilename = '../dat/SG/k_dust/' + filename + '_loss_test.npz'
    np.savez_compressed(paramfilename, model = model, param = param2, prof = prof2, \
        time = time, Ecool = Ecool, Eevap = Eevap, i = i, time2 = time2)                  
        
    return prof2, param2, time, Ecool, Eevap, i, flag, time2
    
#def mass_loss_tweak_n(filename, prms, td = 3e6, tol = 1e-24, ni = 10, nf = 200, nMpoints = 5000):
#    
#    #prof1, param1, time1, Ecool1, Eevap1, i1, flag = mass_loss(filename, prms, td, tol, ni, nMpoints)
#    prof2, param2, time, Ecool, Eevap, i, flag = mass_loss(filename, prms, td, tol, nf, nMpoints)
#    
#    while flag == 1:
#        n = (ni + nf) / 2
#        prof2, param2, time, Ecool, Eevap, i, flag = mass_loss(filename, prms, td, tol, n, nMpoints)
                      
            
#    
#    
#    
#    i = 1
#    
#    while(np.abs(Ecool - Eevap)) >= 0:
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
    
    
    
    
    #return MBd, Ld, EtotBd, RBd, y
    
    
    
    
    
    
    