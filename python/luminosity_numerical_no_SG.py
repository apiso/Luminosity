"""



Shooting code that takes into account the self-gravity of the radiative zone,
assuming constant luminosity in the radiative zone.

Call Ltop to find the luminosity corresponding to a converged atmosphere for a
given mass; generally L1 = 10**20, L2 = 10**30, as a physical luminosity
values should lie between the two:
    e.g. Ltop(10*Me, 10**20, 10**30, 500, 10**(-25))

Call shoot to find the P, T, M profile of a converged atmosphere with a given
mass.    e.g. shoot(10*Me, 10**20, 10**30, 500, 10**(-25))


"""





from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
    cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk, params, \
    kconst, RB, kdustall
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

#-----------------------------------------------------------------------------

#delad = 2./7
#a = 0.1
#Mc = 10 * Me
#rc = (3*Mc/(4*np.pi*rhoc))**(1./3)            
            
prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
    Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
    Td = Tdisk(a, FT), kappa = kdust)
    
A = 5 * np.pi / 16
mu = 2.35 * mp
gammac = 4./3
muc = 60 * mp


def delradfn(p, m, T, L, prms = prms): #radiative temperature gradient
    
    rho = p / (prms.R * T)
    return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)


def Del(p, m, T, L, prms = prms): #del = min(delad, delrad)

    return min(prms.delad, delradfn(p, m, T, L, prms))


def Ltop(Tc, L1, L2, n, tol, prms = prms):



    """

    Iterates among luminosity values until a converged atmosphere solution is
    found for a given mass Mi.   

    Implements the shooting method to solve for a model atmosphere, including
    self-gravity in both convective and radiative regions. It shoots for the
    radiative luminosity of an atmosphere with a given total mass. The
    atmosphere and disk parameters (prms) can be explicitly specified.

    Input
    ----
    Mi:
        total atmosphere mass (including mass up to the Hill radius) in g
    L1, L2:

        trial luminosity values. L1 and L2 must bracket the solution.
    n:
        number of grid points
    tol:
        relative tolerance for core matching
    prms:
        gas, disk and core parameters. Default used by importing values from
        parameters.py module

    Output
    ------
    Lmatch:
        best fit luminosity
    delta:
        matching error

    """


    L1 = float(L1)
    L2 = float(L2)

    def f(x, r):

        """
        structure eqns. "x" = [p , T , m, L]
        dp/dr = - G * m * P / (r**2 * R * T),
        dT/dr = - del * G * m / (R * r**2)
        #####dm/dr = 4 * pi * r**2 * p / (R * T)
        dL/dr = 0

        """
        
        return np.array([ - G * prms.Mco * x[0] / (r**2 * prms.R * x[1]), \
                            - Del(x[0], prms.Mco, x[1], x[2], prms) * G * prms.Mco / \
                                (prms.R * r**2), \
                            0])              


    def delta(lum):      

        """
        Returns the relative error between the core mass obtained by integrating
        from a luminosity lum and the actual core mass.
        """
        if RB(prms.Mco, prms.a) <= RHill(prms.Mco, prms.a):
            rfit = RB(prms.Mco, prms.a)
        else:
            rfit = RHill(prms.Mco, prms.a)
        #integration of the structure equations
        y = odeint(f, [prms.Pd, prms.Td, lum], [rfit, prms.rco])
        Tc1 = y[:,1][-1] #core mass from the guessed L

        deltaL = 4 / np.pi * np.arctan(Tc1 / Tc) - 1
        #relative error; use of the arctan ensures deltaL stays between -1 and 1
        if math.isnan(deltaL): #used to get rid of possible divergences
                deltaL = 1.

        return deltaL

    Lmatch = brentq(delta, L1, L2, xtol = tol)

    return Lmatch, delta(Lmatch)

###-----------------------------------------------------------------------------


def shoot(Tc, L1, L2, n, tol, prms = prms):
    
    """
    Returns the converged atmospheric profile, first by calling iteration
    routine, then by reintegrating structure equations with virial terms.

    Input
    ----
    Mi:
        total atmosphere mass (including mass up to the Hill radius) in g
    L1, L2:
        trial luminosity values. L1 and L2 must bracket the solution.
    n:
        number of grid points
    tol:
        relative tolerance for core matching
    prms:
        gas, disk and core parameters. Default used by importing values from
        parameters.py module

    Output
    ------
    r, P(r), T(r), m(r), rho(r), delrad(r):
        arrays of length n
    Mtot:
        total atmosphere mass in Earth masses (Mi/Me)
    Mcb:
        total mass at the radiative convective boundary in Earth masses

    MB:
        total mass inside the Bondi radius in Earth masses; this is what we
        define as the mass of the atmosphere
    rcb, RB, RHill:
        radiative convective boundary, Bondi and Hill radii in Earth radii
    Pc, Pcb, PB:
        pressure at the core, RCB and RB
    Tc, Tcb, TB:
        temperature at the core, RCB and RB
    Egcb, Ucb, Etotcb:
        gravitational, internal and total energy at the RCB in ergs
    EgB, UB, EtotB:
        gravitational, internal and total energy at the Bondi radius in ergs
    EgHill, UHill, EtotHill:
        gravitational, internal and total energy at the Hill radius in ergs
    L:
        matched luminosity in ergs/s
    vircb, virHill:
        checks of virial equilibrium at RCB and RHill
    err:
        luminosity matching error

    """

    if RB(prms.Mco, prms.a) <= RHill(prms.Mco, prms.a):
        rfit = RB(prms.Mco, prms.a)
    else:
        rfit = RHill(prms.Mco, prms.a) #sets the outer boundary conditions at the Hill radius
    Lpluserror = Ltop(Tc, L1, L2, n, tol, prms)
    L = Lpluserror[0]
    err = Lpluserror[1]

    def f(x, r):

        """
        structure eqns. "x" = [p , T , m, L, Eg, U, Iu],
            with Iu the 3p/rho dm integral in the virial theorem

        dp/dr = - G * m * P / (r**2 * R * T),
        dT/dr = - del * G * m / (R * r**2)
        ###dm/dr = 4 * pi * r**2 * p / (R * T)
        dL/dr = 0
        dEg/dr = - 4 * pi * G * m * r * P / (R * T)
        dU/dr = 4 * pi * r**2 * P * Cv / R
        dIu/dr = 12 * pi * P * r**2

        """       

        return np.array([ - G * prms.Mco * x[0] / (r**2 * prms.R * x[1]), \
                            - Del(x[0], prms.Mco, x[1], x[2], prms) * G * prms.Mco / \
                            (prms.R * r**2),
                            #4 * pi * r**2 * x[0] / (prms.R * x[1]), \
                            0, \
                            - 4 * pi * G * prms.Mco * r * x[0] / (prms.R * x[1]), \
                            4 * pi * r**2 * x[0] * prms.Cv / prms.R, \
                            12 * pi * r**2 * x[0]]) 

    E0 = G * prms.Mco**2 / rfit
    r = np.logspace(np.log10(rfit), np.log10(prms.rco), n)
        #radius grid
    y = odeint(f, [prms.Pd, prms.Td, L, - E0, E0, E0], r)
        #integration of the structure equations

    #reversess all arrays such that the radius increases from core to Hill
    r = r[::-1]
    P = y[:,0][::-1]
    T = y[:,1][::-1]
    #m = y[:,2][::-1]
    Eg = y[:,3][::-1]
    U = y[:,4][::-1]
    Iu = y[:,5][::-1]

    #setting zero energy point at core
    Eg = Eg - Eg[0] 
    U = U - U[0]
    Iu = Iu - Iu[0]

    delrad = 0 * np.ndarray(shape = len(P), dtype = float)
    for i in range(len(delrad)):
        delrad[i] = delradfn(P[i], prms.Mco, T[i], L, prms)

    rho = P / (prms.R * T)
        
    m = np.zeros(len(r))
    m[0] = prms.Mco
    for i in range(len(r) - 1):
        m[i + 1] = m[i] + 4 * pi * (r[i]**2 * rho[i] + r[i+1]**2 * rho[i+1]) / 2 * (r[i+1] - r[i])

    #interpolation functions to find the RCB
    fr = interp1d(delrad[::-1], r[::-1])
    fP = interp1d(delrad[::-1], P[::-1])
    fT = interp1d(delrad[::-1], T[::-1])
    fm = interp1d(delrad[::-1], m[::-1])
    fEg = interp1d(delrad[::-1], Eg[::-1])
    fU = interp1d(delrad[::-1], U[::-1])
    fIu = interp1d(delrad[::-1], Iu[::-1])

    
    if prms.kappa == kconst:
        try:
            rcb = float(fr(prms.delad))
            Pcb = float(fP(prms.delad))
            Tcb = float(fT(prms.delad))
            Mcb = float(fm(prms.delad))
            Egcb = float(fEg(prms.delad))
            Ucb = float(fU(prms.delad))
            Iucb = float(fIu(prms.delad))
            Etotcb = Egcb + Ucb
        except ValueError:
            rcb = r[-1]
            Pcb = P[-1]
            Tcb = T[-1]
            Mcb = m[-1]
            Egcb = Eg[-1]
            Ucb = U[-1]
            Iucb = Iu[-1]
            Etotcb = Egcb + Ucb
        
    elif prms.kappa == kdust:
        for i in range(len(delrad) - 1):
            if (delrad[i] - prms.delad) * (delrad[i + 1] - prms.delad) < 0:
                break
            rcb = (r[i] + r[i + 1]) / 2
            Pcb = (P[i] + P[i + 1]) / 2
            Tcb = (T[i] + T[i + 1]) / 2
            Mcb = (m[i] + m[i + 1]) / 2
            Egcb = (Eg[i] + Eg[i + 1]) / 2
            Ucb = (U[i] + U[i + 1]) / 2
            Iucb = (Iu[i] + Iu[i + 1]) / 2
            Etotcb = Egcb + Ucb
        if i == len(delrad) - 2:
            rcb = r[-1]
            Pcb = P[-1]
            Tcb = T[-1]
            Mcb = m[-1]
            Egcb = Eg[-1]
            Ucb = U[-1]
            Iucb = Iu[-1]
            Etotcb = Egcb + Ucb
            
        
    Egout = Eg[-1]
    Uout = U[-1]
    Iuout = Iu[-1]
    Etotout = Egout + Uout

    Pc = P[0]
    Tc = T[0]

    #m[-1] = m[-2]

    vircb = (Egcb + Iucb + 4 * pi * (prms.rco**3 * Pc - rcb**3 * Pcb)) / (-Egcb)
    virout = (Egout + Iuout + 4 * pi * (prms.rco**3 * Pc - rfit**3 * prms.Pd)) \
                / (-Egout)

    

    return r, P, T, rho, m, delrad, Eg, U, Mcb / Me, m[-1] / Me, rcb / Re, \
            rfit / Re, Pc, Pcb, Tc, Tcb, Egcb, Ucb, Etotcb, \
            Egout, Uout, Etotout, L, err




def Tcmin(prms):
    
    rfit = min(RB(prms.Mco, prms.a), RHill(prms.Mco, prms.a))
    return prms.Td * (1 + prms.delad * RB(prms.Mco, prms.a) / prms.rco - \
        prms.delad * RB(prms.Mco, prms.a) / rfit)

#--------------------------------------------------------------------------

    

#    dRBondi = r - G * m / (prms.R * prms.Td)
#
#    #r - G m(r)/(R delad) = 0 at RB, so dRBondi(r) = 0 gives RB
#
#    
#
#    if dRBondi[-1] > 0: #ensures RB < RHill
#
#        fRBondi = interp1d(dRBondi, m)
#
#        MB = fRBondi(0)
#
#        RB = (G * MB) / (prms.R * prms.Td)
#
#        fPB = interp1d(m, P)
#
#        fTB = interp1d(m, T)
#
#        fEgB = interp1d(m, Eg)
#
#        fUB = interp1d(m, U)
#
#
#
#        PB = float(fPB(MB))
#
#        TB = float(fTB(MB))
#
#        EgB = float(fEgB(MB))
#
#        UB = float(fUB(MB))
#
#        EtotB = EgB + UB
#
#    else: #if RB > RHill, we are outside our boundaries, so set all Bondi
#
#          #values to the Hill values 
#
#        MB, RB, PB, TB, EgB, UB, EtotB = \
#            Mi, rfit, prms.Pd, prms.Td, EgHill, UHill, EtotHill


    







            

    

