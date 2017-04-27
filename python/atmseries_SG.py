
"""
The script generates series of atmospheres for varying core masses at a fixed distance in the disk.
"""

import numpy
from scipy import integrate
from scipy.interpolate import interp1d
from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk
from utils.parameters import mstar, FSigma, FT
from utils.constants import params
from RadSGPoly.profiles_poly import profiles_write
from RadSGPoly.cooling_poly import critical

def atmseries(a, rhoc, delad, Y, Mcomin, Mcomax, nMco, n = 500, \
              nMpoints = 300, L1 = 10**17, L2 = 10**28, minMfrac = 1.1, \
              maxMfrac = 2.8, opacity = kdust):
    
    """

    Generates atmosphere profiles for a range of core masses at fixed distance
    in the disk.

    Input
    -----
    a:
        distance in the disk in AU
    rhoc:
        core density in g/cm^3; default value used rhoc = 3.2
    delad:
        adiabatic gradient
    Y:
        Helium core mass fraction
    Mcomin:
        minimum core mass for which we want to generate an atmosphere
    Mcomax:
        maximum core mass for which we want to generate an atmosphere
    nMco:
        number of core masses for which we want to generate an atmosphere
    n:
        number of points for the profile of one atmosphere of fixed mass.
        The default is 500.
    nMpoints:
        number of atmosphere profiles of fixed mass for a given core mass.
        The default is 300.
    L1, L2:
        luminosities that bracket the matched solution in erg/s.
        Default L1 = 10**21, L2 = 10**30
    minMfrac:
        fractional mass (expressed in core masses) of the lowest mass
        atmosphere (including core mass) for a given core. Default 1.1
    maxMfrac:
        fractional mass (expressed in core masses) of the highest mass
        atmosphere (including core mass) for a given core. Default 1.1


    Returns
    -------
    .npz files containing atmosphere profiles
    
    """

    #setting up gas and disk parameters
    gamma = gammafn(delad)
    R = Rfn(Y)
    Cv = Cvfn(Y, delad)
    Pd = Pdisk(a, mstar, FSigma, FT)
    Td = Tdisk(a, FT)

    Mcore = numpy.linspace(Mcomin, Mcomax, nMco) #probably linear spacing is
                            #the best to use, but can be changed
    rcore = (3 * Mcore / (4 * numpy.pi * rhoc))**(1./3)

    for i in range(nMco):
        
        prms = params(Mcore[i], rcore[i], a, delad, Y, gamma, R, Cv, Pd, Td,\
                      kappa = opacity)
        Mmin = minMfrac * Mcore[i]
        Mmax = maxMfrac * Mcore[i]

        """
        There are cases in which Mmin may be too low, so we use a try-except
        clause. If Mmin is too low (gives a value error), we keep increasing
        Mmin by small amounts (default 1.05 * previous Mmin) until we find a
        solution.
        """

        success = 0

        while success != 1:
        
            try:
                profiles_write(n, nMpoints, L1, L2, Mmin, Mmax, 'Mc' + \
                               str(Mcore[i]/Me)[:4], prms = prms)
                success = 1
                
            except ValueError:
                Mmin = Mmin * 1.05

        print "SUCCESS FOR MC = %s !" % str(Mcore[i]/Me)



def mcrit_disktime(Mcseries, tseries, disklife = (365 * 24 * 3600)):

    """

    Interpolates to find the critical core mass to form an atmosphere within a
    given disk lifetime.

    Input
    -----
    Mcseries:
        list or array of core masses
    tseries:
        list or array of pre-calculated cooling times for Mcseries
    disklife:
        disk lifetime in years; default 3Myrs


    Output
    ------
    mcrit:
        critical core mass in Earth masses
    
    """
    
    ft = interp1d(tseries[::-1], Mcseries[::-1])
    mcrit = float(ft(disklife))

    return mcrit
    
        

    
