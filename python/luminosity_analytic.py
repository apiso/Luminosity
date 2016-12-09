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
from sympy import solveset

#prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
#    Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
#    Td = Tdisk(a, FT), kappa = 0.1)    

delad = 2./7
a = 0.1
Mc = 10 * Me
rc = (3*Mc/(4*np.pi*rhoc))**(1./3)            
            
prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
    Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
    Td = 1e3, kappa = 0.1)
    
A = 5 * np.pi / 16
mu = 2.35 * mp
gammac = 4./3
muc = 60 * mp
    

def RB(prms):
    
    return G * prms.Mco / (prms.R * prms.Td)
    
def RBp(prms):
    
    return RB(prms) * prms.delad
    
def Trcb(prms):
    
    return (prms.gamma - 1) / prms.gamma * G * prms.Mco * mu / (RBp(prms) * kb)
        
def rhorcb(Matm, Rrcb, prms):
    
    return Matm * (RBp(prms) / Rrcb)**(1 / (1 - prms.gamma)) / (4 * A * np.pi * Rrcb**3)
    
def E(Matm, Rrcb, prms):
    
    return - (prms.gamma - 1)**2 / (A * prms.gamma * (3 - 2 * prms.gamma)) * \
        G * prms.Mco * Matm / prms.rco * \
            (Rrcb / prms.rco)**(- (3 * prms.gamma - 4) / (prms.gamma - 1))
            
def L(Matm, Rrcb, prms):
    
    return 64 * np.pi / 3 * sigma * Trcb(prms)**4 * RBp(prms) / \
        (prms.kappa * rhorcb(Matm, Rrcb, prms))
        
def time(Matm, Rrcb, prms):
    
    return - E(Matm, Rrcb, prms) / L(Matm, Rrcb, prms)

def Matm(t, Rrcb, prms):
    
    def f(x):
        return time(x, Rrcb, prms) - t
        
    return brentq(f, 1, 100 * Me)    
            
def Luminosity(t, prms):
    
    Rrcb = RBp(prms)
    
    return L(Matm(t, Rrcb, prms), Rrcb, prms)
    
###############################################################################

def deltatevap(prms, td = 3e6 * yr):
    
    return td * (RBp(prms) / prms.rco)**(- (3 - 2 * prms.gamma) / (prms.gamma - 1)) * \
        A * prms.gamma * (3 - 2 * prms.gamma) / (prms.gamma - 1)**2
        
def Matmevap(prms, td = 3e6 * yr):
    
    Rrcb = RBp(prms)
    M = Matm(td, Rrcb, prms)
    
    rhorcbevap = rhorcb(M, Rrcb, prms)
    Rrcbevap = prms.rco
    
    return A * 4 * np.pi * Rrcbevap**3 * rhorcbevap * \
        (RBp(prms)/Rrcbevap)**(1 / (prms.gamma - 1))   
    
###############################################################################

def tshrink(Rrcb, prms, td = 3e6 * yr):
    
    Matm = Matmevap(prms, td)
    
    return 1 / (256 * np.pi**2) * 3 * prms.gamma**2 / (prms.gamma - 1) * \
        (kb / mu) * Matm / (sigma * prms.Td**3 * prms.rco**4) * \
            (RBp(prms) * Rrcb / prms.rco**2)**(-1 / (prms.gamma - 1)) * \
                (prms.gamma / (2 * prms.gamma - 1) * Matm + \
                    1/prms.gamma * (prms.gamma - 1) / (gammac - 1) * mu / muc * prms.Mco)
            
  
def Rrcbshrink(t, prms, td = 3e6 * yr):
    
    def f(x):
        return tshrink(x, prms, td) - t
        
    return brentq(f, 1, 1e5*RB(prms))


def rhorcbshrink(t, prms, td = 3e6 * yr):
    
    Matm = Matmevap(prms, td)
    Rrcb = Rrcbshrink(t, prms, td)
    
    return Matm * prms.gamma / (prms.gamma - 1) * 1 / (4 * np.pi * prms.rco**2 * Rrcb) * \
        (prms.rco**2 / (RBp(prms) * Rrcb))**(1 / (prms.gamma - 1))      
        
def Lshrink(t, prms, td = 3e6 * yr):
    
    return 64 * np.pi / 3 * sigma * Trcb(prms)**4 * RBp(prms) / \
        (prms.kappa * rhorcbshrink(t, prms, td))      

def timeshrink(prms, td = 3e6 * yr):
    
    def f(x):
        return Lshrink(x, prms, td) - Luminosity(td, prms)
    return brentq(f, 1, 1e20)
                
def Lshrinknorm(t, prms, td = 3e6 * yr):
    
        return Lshrink(t + td + deltatevap(prms, td), prms, td) - \
            (Lshrink(td + deltatevap(prms, td), prms, td) - Luminosity(td, prms))        
        
        
###############################################################################

def rhorcblight(t, prms, td = 3e6 * yr):
    
    M = Matmevap(prms, td)
    Rrcb = prms.rco
    cs = np.sqrt(prms.R * prms.Td)
    deltaM = 4 * np.pi * RB(prms) * prms.Pd / (prms.R * prms.Td) * cs * (t - td - deltatevap(prms, td))
    
    rho = rhorcb(M - deltaM, Rrcb, prms)
    if rho >= 0:
        return rho
    else:
        return 0
        
def Llight(t, prms):
    
    rho = rhorcblight(t, prms)
    if rho != 0:
        return 64 * np.pi / 3 * sigma * Trcb(prms)**4 * RBp(prms) / \
            (prms.kappa * rho)
    else:
        return 0
    

###############################################################################

def Lglobal(prms, tmin, tmax, npts, td = 3e6 * yr):
            
    t = np.logspace(np.log10(tmin), np.log10(tmax), npts)
    L = np.zeros(npts)
    M = Matmevap(prms, td)
    
    for i in range(npts):
        if t[i] <= td:
            L[i] = Luminosity(t[i], prms)
        elif td < t[i] <= td + deltatevap(prms, td):
            L[i] = Luminosity(td, prms)
        else:
            if M / prms.Mco >= mu / muc:
                L[i] = Lshrink(t[i], prms, td) / \
                    (Lshrink(td + deltatevap(prms, td), prms, td) / Luminosity(td, prms)) #- \
                #(Lshrink(td + deltatevap(prms, td), prms, td) - Luminosity(td, prms)) #Lshrinknorm(t[i] - td, prms, td)
            else:
                L[i] = Llight(t[i], prms)
    return t, L
    
        