"""

Shooting code that takes into account the self-gravity of the radiative zone,
assuming constant luminosity in the radiative zone.

Call Ltop to find the luminosity corresponding to a converged atmosphere for a
given mass; generally L1 = 10**20, L2 = 10**30, as a physical luminosity
values should lie between the two:
    e.g. Ltop(10*Me, 10**20, 10**30, 500, 10**(-25))

Call shoot to find the P, T, M profile of a converged atmosphere with a given
mass.
    e.g. shoot(10*Me, 10**20, 10**30, 500, 10**(-25))

'shoot' is used in profiles_poly to generate recarrays and .npz files of model
atmospheres

"""


from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, kdust10, kdustbeta1, \
     Tdisk, Pdisk, kfixed, kdustall, \
     paramsEOS
from utils.parameters import FT, FSigma, mstar, Y, delad, rhoc, Mc, rc, \
     gamma, Y, a #, R_EOS
from utils.interpolation_functions import interplog10rho, interpdelad, \
     interplog10u
import numpy
from numpy import pi
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
from scipy import integrate, interpolate, optimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import brentq, brenth, bisect, newton, fminbound
from gg_opacity import interp_opacity, kR

#-----------------------------------------------------------------------------

prms = paramsEOS(Mc, rc, Y, a, Pd = Pdisk(a, mstar, FSigma, FT), \
                 Td = Tdisk(a, FT), kappa = kdust) #gas, disk and core parameters
                        #for specific values imported from parameters.py 

def delradfn(p, m, T, L, prms = prms): #radiative temperature gradient
    if prms.kappa == kdust:
        return 3 * prms.kappa(T) * p * L / (64 * pi * G * m * sigma * T**4)
    elif prms.kappa == kfixed:
        return 3 * prms.kappa() * p * L / (64 * pi * G * m * sigma * T**4)
    elif prms.kappa == kdustbeta1:
        return 3 * prms.kappa(T, prms.a) * p * L / (64 * pi * G * m * sigma * T**4)
    elif prms.kappa == kdustall:
        rho = 10**interplog10rho(numpy.log10(T), numpy.log10(p))
        return 3 * prms.kappa(T, rho) * p * L / (64 * pi * G * m * sigma * T**4)
    elif prms.kappa == interp_opacity:
        if T <= 1000:
            return 3 * prms.kappa(T) * p * L \
                   / (64 * pi * G * m * sigma * T**4)
        else:
            rho = 10**interplog10rho(numpy.log10(T), numpy.log10(p))
            return 3 * min(prms.kappa(T), kdustall(T, rho)) * p * L \
                   / (64 * pi * G * m * sigma * T**4)

def Del(p, m, T, L, delad, prms = prms): #del = min(delad, delrad)
    return min(delad, delradfn(p, m, T, L, prms))

def Ltop(Mi, L1, L2, n, tol, prms = prms, checktop = 0):

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
    checktop:
        flag to study EOS effects separately
            if 0, pure EOS throughout
            if 1, atmosphere top is a polytrope
            if -1, atmosphere bottom is a polytrope


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
        dp/dr = - G * m * rho / (r**2),
        dT/dr = - del *  T * rho * G * m / (P * r**2)
        dm/dr = 4 * pi * r**2 * rho
        dL/dr = 0
        """

        if checktop == 0:
    
            rho = 10**interplog10rho(numpy.log10(x[1]), numpy.log10(x[0]))
            delad = interpdelad(numpy.log10(x[1]), numpy.log10(x[0]))

        else:
            if checktop * (x[1] - 500.0) > 0:
                rho = 10**interplog10rho(numpy.log10(x[1]), numpy.log10(x[0]))
                delad = interpdelad(numpy.log10(x[1]), numpy.log10(x[0]))
            else:
                delad = 0.29677
                R = Rfn(prms.Y)
                rho = x[0] / (R * x[1])
        
        return numpy.array([ - G * x[2] * rho / (r**2), \
                             - Del(x[0], x[2], x[1], x[3], delad, prms) \
                             * x[1] * rho * G * x[2]  / (x[0] * r**2),
                             4 * pi * r**2 * rho, \
                             0])              

    def delta(lum):
        
        """
        Returns the relative error between the core mass obtained by integrating
        from a luminosity lum and the actual core mass.

        """
        rfit = RHill(Mi, prms.a)
        r = numpy.logspace(numpy.log10(rfit), numpy.log10(prms.rco), n)
        #integration of the structure equations
        y = odeint(f, [prms.Pd, prms.Td, Mi, lum], r)
            
        Mc1 = y[:,2][-1] #core mass from the guessed L

        deltaL = 4 / numpy.pi * numpy.arctan(Mc1 / prms.Mco) - 1
        #deltaL = 1 - prms.Mco / Mc1

        if math.isnan(deltaL): #used to get rid of possible divergences
            
            if lum < 10**25:
            	deltaL = 1. *(-1)
	    else:
		deltaL = 1.
        return deltaL

    Lmatch = brentq(delta, L1, L2, xtol = tol, maxiter = 500)

    return Lmatch, delta(Lmatch)



  
###-----------------------------------------------------------------------------

def shoot(Mi, L1, L2, n, tol, prms = prms, checktop = 0):

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
    checktop:
        flag to study EOS effects separately
            if 0, pure EOS throughout
            if 1, atmosphere top is a polytrope
            if -1, atmosphere bottom is a polytrope


    Output
    ------
    r, P(r), T(r), m(r), rho(r), delrad(r), delad(r):
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

    rfit = RHill(Mi, prms.a) #sets the outer boundary conditions at the Hill radius
    Lpluserror = Ltop(Mi, L1, L2, n, tol, prms, checktop)
    L = Lpluserror[0]
    err = Lpluserror[1]

        

    def f(x, r):
        
        """
        structure eqns. "x" = [p , T , m, L, Eg, U, Iu],
            with Iu the 3p/rho dm integral in the virial theorem
        dp/dr = - G * m * rho / (r**2),
        dT/dr = - del *  T * rho * G * m / (P * r**2)
        dm/dr = 4 * pi * r**2 * rho
        dL/dr = 0
        dEg/dr = - 4 * pi * G * r * m * rho
        dU/dr = 4 * pi * r**2 * rho * u
        dIu/dr = 12 * pi * r**2 * P
        """

        if checktop == 0:
    
            rho = 10**interplog10rho(numpy.log10(x[1]), numpy.log10(x[0]))
            delad = interpdelad(numpy.log10(x[1]), numpy.log10(x[0]))
            u = 10**interplog10u(numpy.log10(x[1]), numpy.log10(x[0]))

        else:
            if checktop * (x[1] - 500.0) > 0:
                rho = 10**interplog10rho(numpy.log10(x[1]), numpy.log10(x[0]))
                delad = interpdelad(numpy.log10(x[1]), numpy.log10(x[0]))
                u = 10**interplog10u(numpy.log10(x[1]), numpy.log10(x[0]))
                
            else:
                delad = 0.29677
                R = Rfn(prms.Y)
                Cv = Cvfn(prms.Y, delad)
                rho = x[0] / (R * x[1])
                u = Cv * x[1]
        
        return numpy.array([ - G * x[2] * rho / (r**2), \
                             - Del(x[0], x[2], x[1], x[3], delad, prms) \
                             * x[1] * rho * G * x[2]  / (x[0] * r**2),
                             4 * pi * r**2 * rho, \
                             0,\
                             - 4 * numpy.pi * G * r * x[2] * rho,\
                             4 * numpy.pi * r**2 * rho * u,\
                             12 * numpy.pi * r**2 * x[0]])
    
    E0 = G * Mi**2 / rfit
    r = numpy.logspace(numpy.log10(rfit), numpy.log10(prms.rco), n)
        #radius grid
    y = odeint(f, [prms.Pd, prms.Td, Mi, L, - E0, E0, E0], r)
        #integration of the structure equations

    #reversess all arrays such that the radius increases from core to Hill
    r = r[::-1]
    P = y[:,0][::-1]
    T = y[:,1][::-1]
    m = y[:,2][::-1]
    Eg = y[:,4][::-1]
    U = y[:,5][::-1]
    Iu = y[:,6][::-1]

    #setting zero energy point at core
    Eg = Eg - Eg[0] 
    U = U - U[0]
    Iu = Iu - Iu[0]

    delrad = 0 * numpy.ndarray(shape = len(P), dtype = float)
    for i in range(len(delrad)):
        delrad[i] = delradfn(P[i], m[i], T[i], L, prms)
    #delrad = delradfn(P, m, T, L, prms)
    
    if checktop == 0:
        delad = interpdelad(numpy.log10(T), numpy.log10(P))

        fdelad = interp1d(delrad[::-1] - delad[::-1], delad[::-1])
        deladcb = float(fdelad(0))
    
        rho = 10**interplog10rho(numpy.log10(T), numpy.log10(P))
        u = 10**interplog10u(numpy.log10(T), numpy.log10(P))



    else:

        #fTr = interp1d(T[::-1], r[::-1])
        #fTP = interp1d(T[::-1], P[::-1])
        #fT = interp1d(delrad[::-1], T[::-1])
        #fTm = interp1d(T[::-1], m[::-1])
        #fTEg = interp1d(T[::-1], Eg[::-1])
        #fTU = interp1d(delrad[::-1], U[::-1])
        #fTIu = interp1d(delrad[::-1], Iu[::-1])


        deladcb = 0.29677
        delad = numpy.array([deladcb] * n)
        R = Rfn(prms.Y)
        Cv = Cvfn(prms.Y, delad)
        
        rho = P / (R * T)
        u = Cv * T

    
    #interpolation functions to find the RCB
    fr = interp1d(delrad[::-1], r[::-1])
    fP = interp1d(delrad[::-1], P[::-1])
    fT = interp1d(delrad[::-1], T[::-1])
    fm = interp1d(delrad[::-1], m[::-1])
    fEg = interp1d(delrad[::-1], Eg[::-1])
    fU = interp1d(delrad[::-1], U[::-1])
    fIu = interp1d(delrad[::-1], Iu[::-1])

    if prms.kappa != interp_opacity:

        rcb = float(fr(deladcb))
        Pcb = float(fP(deladcb))
        Tcb = float(fT(deladcb))
        Mcb = float(fm(deladcb))
        Egcb = float(fEg(deladcb))
        Ucb = float(fU(deladcb))
        Iucb = float(fIu(deladcb))
        Etotcb = Egcb + Ucb

    else:
        deldif = delad - delrad
        rcbivec = []
        for i in range(len(r) - 1):
            if (delad[i] - delrad[i]) * (delad[i + 1] - delrad[i + 1]) <=0:
                rcbivec = numpy.append(rcbivec, i)

        delcb1 = []
        delcb2 = []

        rcb1, Pcb1, Tcb1, Mcb1, Egcb1, Ucb1, Iucb1 = \
              [], [], [], [], [], [], []
        rcb2, Pcb2, Tcb2, Mcb2, Egcb2, Ucb2, Iucb2 = \
              [], [], [], [], [], [], []
        
        for i in range(len(rcbivec)):
            
            if deldif[rcbivec[i]] <= deldif[rcbivec[i] + 1]:
                
                fdel = interp1d(deldif[rcbivec[i]:rcbivec[i] + 2], \
                                delad[rcbivec[i]:rcbivec[i] + 2])
                temp = float(fdel(0))
                delcb1 = numpy.append(delcb1, temp)

                if delrad[rcbivec[i]] <= delrad[rcbivec[i] + 1]:

                    fr = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  r[rcbivec[i]:rcbivec[i] + 2])
                    fP = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  P[rcbivec[i]:rcbivec[i] + 2])
                    fT = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  T[rcbivec[i]:rcbivec[i] + 2])
                    fm = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  m[rcbivec[i]:rcbivec[i] + 2])
                    fEg = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  Eg[rcbivec[i]:rcbivec[i] + 2])
                    fU = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  U[rcbivec[i]:rcbivec[i] + 2])
                    fIu = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  Iu[rcbivec[i]:rcbivec[i] + 2])
                else:
                
                    fr = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  r[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fP = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  P[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fT = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  T[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fm = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  m[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fEg = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  Eg[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fU = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  U[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fIu = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  Iu[rcbivec[i]:rcbivec[i] + 2][::-1])
                
                rcb1 = numpy.append(rcb1, float(fr(temp)))
                Pcb1 = numpy.append(Pcb1, float(fP(temp)))
                Tcb1 = numpy.append(Tcb1, float(fT(temp)))
                Mcb1 = numpy.append(Mcb1, float(fm(temp)))
                Egcb1 = numpy.append(Egcb1, float(fEg(temp)))
                Ucb1 = numpy.append(Ucb1, float(fU(temp)))
                Iucb1 = numpy.append(Iucb1, float(fIu(temp)))

                
            else:
                
                fdel = interp1d(deldif[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                delad[rcbivec[i]:rcbivec[i] + 2][::-1])
                temp = float(fdel(0))
                delcb2 = numpy.append(delcb2, temp)

                if delrad[rcbivec[i]] <= delrad[rcbivec[i] + 1]:
                
                    fr = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  r[rcbivec[i]:rcbivec[i] + 2])
                    fP = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  P[rcbivec[i]:rcbivec[i] + 2])
                    fT = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  T[rcbivec[i]:rcbivec[i] + 2])
                    fm = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  m[rcbivec[i]:rcbivec[i] + 2])
                    fEg = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  Eg[rcbivec[i]:rcbivec[i] + 2])
                    fU = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  U[rcbivec[i]:rcbivec[i] + 2])
                    fIu = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2], \
                                  Iu[rcbivec[i]:rcbivec[i] + 2])
                else:

                    fr = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  r[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fP = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  P[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fT = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  T[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fm = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  m[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fEg = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  Eg[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fU = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  U[rcbivec[i]:rcbivec[i] + 2][::-1])
                    fIu = interp1d(delrad[rcbivec[i]:rcbivec[i] + 2][::-1], \
                                  Iu[rcbivec[i]:rcbivec[i] + 2][::-1])                    
                
                rcb2 = numpy.append(rcb2, float(fr(temp)))
                Pcb2 = numpy.append(Pcb2, float(fP(temp)))
                Tcb2 = numpy.append(Tcb2, float(fT(temp)))
                Mcb2 = numpy.append(Mcb2, float(fm(temp)))
                Egcb2 = numpy.append(Egcb2, float(fEg(temp)))
                Ucb2 = numpy.append(Ucb2, float(fU(temp)))
                Iucb2 = numpy.append(Iucb2, float(fIu(temp)))

        delcb = numpy.append(delcb1, delcb2)
        
        rcb = numpy.append(rcb1, rcb2)
        Pcb = numpy.append(Pcb1, Pcb2)
        Tcb = numpy.append(Tcb1, Tcb2)
        Mcb = numpy.append(Mcb1, Mcb2)
        Egcb = numpy.append(Egcb1, Egcb2)
        Ucb = numpy.append(Ucb1, Ucb2)
        Iucb = numpy.append(Iucb1, Iucb2)


        for i in range(len(rcb) - 1):
            if rcb[i] < rcb[i + 1]:
                pass
            else:
                temp = rcb[i]
                rcb[i] = rcb[i + 1]
                rcb[i + 1] = temp

                temp = Pcb[i]
                Pcb[i] = Pcb[i + 1]
                Pcb[i + 1] = temp

                temp = Tcb[i]
                Tcb[i] = Tcb[i + 1]
                Tcb[i + 1] = temp

                temp = Mcb[i]
                Mcb[i] = Mcb[i + 1]
                Mcb[i + 1] = temp

                temp = Egcb[i]
                Egcb[i] = Egcb[i + 1]
                Egcb[i + 1] = temp

                temp = Ucb[i]
                Ucb[i] = Ucb[i + 1]
                Ucb[i + 1] = temp

                temp = Iucb[i]
                Iucb[i] = Iucb[i + 1]
                Iucb[i + 1] = temp


        Etotcb = Egcb + Ucb

            
    EgHill = Eg[-1]
    UHill = U[-1]
    IuHill = Iu[-1]
    EtotHill = EgHill + UHill

    Pc = P[0]
    Tc = T[0]

    dRBondi = r - G * m / (Rfn(prms.Y) * prms.Td)
    #r - G m(r)/(R delad) = 0 at RB, so dRBondi(r) = 0 gives RB
    
    if dRBondi[-1] > 0: #ensures RB < RHill
        fRBondi = interp1d(dRBondi, m)
        MB = fRBondi(0)
        RB = (G * MB) / (Rfn(prms.Y) * prms.Td)
        fPB = interp1d(m, P)
        fTB = interp1d(m, T)
        fEgB = interp1d(m, Eg)
        fUB = interp1d(m, U)

        PB = float(fPB(MB))
        TB = float(fTB(MB))
        EgB = float(fEgB(MB))
        UB = float(fUB(MB))
        EtotB = EgB + UB
    else: #if RB > RHill, we are outside our boundaries, so set all Bondi
          #values to the Hill values 
        MB, RB, PB, TB, EgB, UB, EtotB = \
            Mi, rfit, prms.Pd, prms.Td, EgHill, UHill, EtotHill

    vircb = (Egcb + Iucb + 4 * pi * (prms.rco**3 * Pc - rcb**3 * Pcb)) / (-Egcb)
    virHill = (EgHill + IuHill + 4 * pi * (prms.rco**3 * Pc - rfit**3 * prms.Pd)) \
              / (-EgHill)
    
    return r, P, T, m, rho, delrad, delad, Mi / Me, Mcb / Me, MB / Me, rcb / Re, \
           RB / Re, rfit / Re, Pc, Pcb, PB, Tc, Tcb, TB, Egcb, Ucb, Etotcb, \
           EgB, UB, EtotB, EgHill, UHill, EtotHill, L, vircb, virHill, err


#--------------------------------------------------------------------------
    
def mass_conv(M1, M2, n, tol, Y, a, Mc):

    """
    Finds the mass of the fully convective solution (i.e. minimum atmosphere mass)
    
    """

    rc = (3 * Mc / (4 * numpy.pi * rhoc))**(1./3)

    prms = paramsEOS(Mc, rc, Y, a, Pd = Pdisk(a, mstar, FSigma, FT), \
                 Td = Tdisk(a, FT), kappa = kdust)

    def f(x, r):
        """
        structure eqns. "x" = [p , T , m, L]
        dp/dr = - G * m * rho / (r**2),
        dT/dr = - del *  T * rho * G * m / (P * r**2)
        dm/dr = 4 * pi * r**2 * rho
        """

        rho = 10**interplog10rho(numpy.log10(x[1]), numpy.log10(x[0]))
        delad = interpdelad(numpy.log10(x[1]), numpy.log10(x[0]))

        
        return numpy.array([ - G * x[2] * rho / (r**2), \
                             - delad * x[1] * rho * G * x[2]  / (x[0] * r**2),
                             4 * pi * r**2 * rho])              

    def delta(mass):
        
        """
        Returns the relative error between the core mass obtained by integrating
        from a luminosity lum and the actual core mass.

        """
        rfit = RHill(mass, prms.a)
        r = numpy.logspace(numpy.log10(rfit), numpy.log10(prms.rco), n)
        #integration of the structure equations
        y = odeint(f, [prms.Pd, prms.Td, mass], r)
            
        Mc1 = y[:,2][-1] #core mass from the guessed L

        deltam = 4 / numpy.pi * numpy.arctan(Mc1 / prms.Mco) - 1
	
        return deltam

    Mmatch = brentq(delta, M1, M2, xtol = tol, maxiter = 500)

    return Mmatch / Me - Mc / Me, delta(Mmatch)




            
def deltafn(mass, n, Y, a, Mc):

    rc = (3 * Mc / (4 * numpy.pi * rhoc))**(1./3)

    prms = paramsEOS(Mc, rc, Y, a, Pd = Pdisk(a, mstar, FSigma, FT), \
                 Td = Tdisk(a, FT), kappa = kdust)



    def f(x, r):

##       
##    """
##    structure eqns. "x" = [p , T , m, L]
##    dp/dr = - G * m * rho / (r**2),
##    dT/dr = - del *  T * rho * G * m / (P * r**2)
##    dm/dr = 4 * pi * r**2 * rho
##    """

    
        rho = 10**interplog10rho(numpy.log10(x[1]), numpy.log10(x[0]))
        delad = interpdelad(numpy.log10(x[1]), numpy.log10(x[0]))

        
        return numpy.array([ - G * x[2] * rho / (r**2), \
                             - delad * x[1] * rho * G * x[2]  / (x[0] * r**2),
                             4 * pi * r**2 * rho]) 
        
    """
    Returns the relative error between the core mass obtained by integrating
    from a luminosity lum and the actual core mass.

    """
    rfit = RHill(mass, prms.a)
    r = numpy.logspace(numpy.log10(rfit), numpy.log10(prms.rco), n)
    #integration of the structure equations
    y = odeint(f, [prms.Pd, prms.Td, mass], r)
        
    Mc1 = y[:,2][-1] #core mass from the guessed L

    deltam = 4 / numpy.pi * numpy.arctan(Mc1 / prms.Mco) - 1
    
    return - deltam



def Mcoremax(n, Y, a, Mc1, Mc2, Mmax = 100*Me):
    

    def errcore(Mc):
        mopt, deltamax, ierr, numfunc = fminbound(\
            deltafn, Mc, 100*Me, args = (n, Y, a, Mc), full_output = 1)

        return deltamax

    Mcmax = brentq(errcore, Mc1, Mc2)

    return Mcmax, errcore(Mcmax)





    


            
    
