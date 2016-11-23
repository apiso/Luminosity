"""

Constants for atmosphere modelling ; cgs unless otherwise noted

"""
import numpy as np
from collections import namedtuple

#physical
G = 6.673e-08
c = 2.99792458e10
h = 6.626068e-27
kb = 1.3807e-16
mp = 1.67262171e-24
mH = mp
NA = 6.02214129e23
Rb = 83144720.0 #NA*kb #ideal gas constant (per mol not per H!)
RH = Rb/2
RHe = Rb/4
sigma =  5.670373e-5

#unit conversion
cmperau = 14959788700000.0
AU = 1.49597887e13
yr = 365*24*3600
Myr = 3.16e13
barP = 1e6 #dyne/cm*2 to bar (not that `bar` in matplotlib is problem in pylab)

#astronomical
Me = 5.9742e27
Re = 6.371e8
Msun = 1.9891e33
Rsun = 6.9551e10

def RHill(mp, a, Ms = Msun):
    """
    Hill radius in cgs with SMA (`a`) in AU and planet mass (`mp`) also
    in cgs
    """
    return (mp / 3 / Ms)**(1./3) * a * AU

#MMSN disk model
#(could/ should make the deviations from standard MMSN optional or keyword parameters)
def Tdisk(a, FT):  
    """Disk temperature as function of semi-major axis (a in AU) and Ft"""
    return 120 * FT * (a)**(-3./7)

def Pdisk(a, mstar, FSigma, FT): 
    """
    Disk pressure (in dyn cm^-2) as function of semi-major axis (a in AU),
    stellar mass (mstar in solar masses), and factors Fsigma and Ft
    """
    return 1.1 * 10**(-4) * FSigma * np.sqrt(FT * mstar) * a**(-45./14) * 10**6

def Hdisk(a, FT):  #a in AU
    """Disk scaleheight (in cm) as function of semi-major axis and Ft"""
    return 0.021784 * AU * np.sqrt(FT) * (a)**(9./7) 

#opacity laws

def kdust(T, rho = None, b = 2, fsolar = 1.):
    """Opacity (generally applicable for beta = 2)"""
    return 2 * fsolar * (T/100.)**b

def kdustall(T, rho):
    T = float(T)
    rho = float(rho)
    """Full Bell & Lin opacity"""
    if T <= 166.81:
        return 2 * 10**(-4) * T**2
    elif T >= 166.81 and T <= 201.677:
        return 2 * 10**16 * T**(-7)
    elif T >= 201.677 and 0.1 * T**(1./2) - 2 * 10**81 * rho * T**(-24) <=0:
        return 0.1 * T**(1./2)
    elif 0.1 * T**(1./2) - 2 * 10**81 * rho * T**(-24) >=0 \
         and 2 * 10**81 * rho * T**(-24) - 10**(-8) * rho**(2./3) * T**3 >= 0:
        return 2 * 10**81 * rho * T**(-24)
    elif 2 * 10**81 * rho * T**(-24) - 10**(-8) * rho**(2./3) * T**3 <=0 \
         and 10**(-8) * rho**(2./3) * T**3 - 10**(-36) * rho**(1./3) * T**10 >= 0:
        return 10**(-8) * rho**(2./3) * T**3
    elif 10**(-8) * rho**(2./3) * T**3 - 10**(-36) * rho**(1./3) * T**10 <= 0 \
         and 10**(-36) * rho**(1./3) * T**10 - 1.5 * 10**20 * rho * T**(-5./2) <= 0:
        return 10**(-36) * rho**(1./3) * T**10
    elif 10**(-36) * rho**(1./3) * T**10 - 1.5 * 10**20 * rho * T**(-5./2) >= 0 \
         and 1.5 * 10**20 * rho * T**(-5./2) - 0.348 >= 0:
        return 1.5 * 10**20 * rho * T**(-5./2) 
    else:
         return 0.348

def kdustallfn(T, rho):
    kappa = 0 * np.ndarray(shape = len(rho), dtype = float)
    for i in range(len(kappa)):
        kappa[i] = kdustall(T, rho[i])
    return kappa

def kdustbeta1(T, a = None, FT = 1):
    return 0.1 * (T/100.0)

def kdust10(T, P = None, b = 2, fsolar = 1.):
    """Opacity reduced by a factor of 10"""
    return 2 * fsolar * (T/100.)**b / 10

def kdust100(T, P = None, b = 2, fsolar = 1.):
    """Opacity reduced by a factor of 100"""
    return 2 * fsolar * (T/100.)**b / 100

def kfixed(T = None, P = None, const = 0.24):
    """a silly function but useful if kappa functions expects a temperature."""
    return const

def gammafn(delad):
    """Polytropic index gamma as function of delad."""
    return 1. /(1 - delad) 

#R, Cv, molecular weight for a given H/He mixture fraction Y

def mufn(Y):
    return 4./(2-Y)

def Rfn(Y):
    return Rb / mufn(Y)

def Cvfn(Y, delad):
    """
    Assumes that the He fraction only affects mu, but not delad.
    """
    return Rfn(Y) / (gammafn(delad) - 1)

#tuple of gas, disk and core parameters for polytrope
params = namedtuple( \
    'params', 'Mco, rco, a, delad, Y, gamma, R, Cv, Pd, Td, kappa')

#tuple of gas, disk and core parameters for real EOS
paramsEOS = namedtuple( \
    'paramsEOS', 'Mco, rco, Y, a, Pd, Td, kappa')

#tuple of gas, disk and core parameters for polytrope with two delad
paramsvar = namedtuple( \
    'params', 'Mco, rco, a, delad1, delad2, Y, gamma1, gamma2, R, Cv1, Cv2, Pd, Td, kappa')



