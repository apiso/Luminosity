"""

Generates series of atmosphere profiles for a set of gas, disk and core
conditions and writes them in .npz files. Also contains functions to load
profiles.

"""



from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk, params, \
     kdust, kdustbeta1, kdust10, kdust100, kconst
from utils.parameters import FT, FSigma, mstar, Y, delad, rhoc, Mc, rc, \
    gamma, Y, a
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from types import FunctionType as function
from collections import namedtuple
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import fminbound, brentq
from luminosity_numerical_no_SG import Ltop, shoot, prms, Tcmin


delad = 2./7
a = 0.1
Mc = 10 * Me
rc = (3*Mc/(4*np.pi*rhoc))**(1./3)            
            
prms = params(Mc, rc, a, delad, Y, gamma = gammafn(delad), R = Rfn(Y), \
    Cv = Cvfn(Y, delad), Pd = Pdisk(a, mstar, FSigma, FT), \
    Td = 1e3, kappa = kconst)
    
A = 5 * np.pi / 16
mu = 2.35 * mp
gammac = 4./3
muc = 60 * mp


def profiles_write(n, nTcpoints, L1, L2, Tcmax, filename, Tcm = Tcmin(prms), prms = prms, \
                   tol = 10**(-25), savefile = 1, disk = 1):


    model = np.array([(prms.Mco, prms.rco, prms.a, prms.delad, prms.Y, \
                          prms.gamma, prms.R, prms.Cv, prms.Pd, prms.Td, \
                          prms.kappa)], \
                        dtype = [('Mco', float), ('rco', float), ('a', float), \
                                 ('delad', float), ('Y', float), \
                                 ('gamma', float), \
                                 ('R', float), ('Cv', float), ('Pd', float), \
                                 ('Td', float), ('kappa', function)])
    
    prof = np.recarray(shape = (nTcpoints, n), \
                          dtype = [('r', float), ('P', float), ('t', float), \
                                   ('rho', float), ('m', float), ('delrad', float), \
                                   ('Eg', float), ('U', float)])
    param = np.recarray(\
        shape = (nTcpoints), \
        dtype = [('Mcb', float), ('Mtot', float), ('rcb', float),\
                 ('rout', float), ('Pc', float), ('Pcb', float),\
                 ('Tc', float), ('Tcb', float), \
                 ('Egcb', float), ('Ucb', float), ('Etotcb', float), \
                 ('Egout', float), ('Uout', float), ('Etotout', float), \
                 ('L', float), ('err', float)])
        

    temp = np.linspace(Tcm, Tcmax, nTcpoints)

    for i in range(nTcpoints):
        sol = shoot(temp[i], L1, L2, n, tol, prms)
        
        param.Mcb[i], param.Mtot[i], param.rcb[i], param.rout[i], \
                       param.Pc[i], param.Pcb[i], param.Tc[i], param.Tcb[i], \
                       param.Egcb[i], param.Ucb[i], param.Etotcb[i], \
                       param.Egout[i], param.Uout[i], param.Etotout[i], param.L[i], \
                       param.err[i] = sol[8:]

        for k in range(n):
            prof.r[i, k], prof.P[i, k], prof.t[i, k], \
                      prof.rho[i, k], prof.m[i, k], prof.delrad[i, k], \
                      prof.Eg[i, k], prof.U[i, k] = \
                      sol[0][k], sol[1][k], sol[2][k], sol[3][k], sol[4][k], \
                      sol[5][k], sol[6][k], sol[7][k]
        print i


    if savefile == 1:
        paramfilename = '../dat/NO_SG/k_constant/' + filename + '.npz'
        np.savez_compressed(paramfilename, model = model, param = param, prof = prof)
            
    
    return model, param, prof


def atmload(filename, prms = prms, disk = 1):
#    if disk == 1:
#        if prms.kappa == kdust or prms.kappa == kdust10:
#            npzdat = numpy.load(userpath + '/dat_ana/MODELS/RadSGPoly/' + 'delad' + \
#                            str(prms.delad)[:4] + '/Y' + str(prms.Y) + '/' + \
#                            str(prms.a) + 'AU/' + filename)
###        elif prms.kappa == kdust10:
###            npzdat = numpy.load(userpath + '/dat_ana/MODELS/RadSGPoly/' + 'delad' + \
###                                str(prms.delad)[:4] + '/Y' + str(prms.Y) + '/kdust10/' + \
###                                str(prms.a) + 'AU/' + filename)
#        elif prms.kappa == kdust100:
#            npzdat = numpy.load(userpath + '/dat_ana/MODELS/RadSGPoly/' + 'delad' + \
#                                str(prms.delad)[:4] + '/Y' + str(prms.Y) + '/' + \
#                                str(prms.a) + 'AU/' + filename)
#        elif prms.kappa == kdustbeta1:
#            npzdat = numpy.load(userpath + '/dat_ana/MODELS/RadSGPoly/' + 'delad' + \
#                                str(prms.delad)[:4] + '/Y' + str(prms.Y) + '/kdustbeta1/' + \
#                                str(prms.a) + 'AU/' + filename)  


    npzdat = np.load('../dat/NO_SG/k_constant/' + filename + '.npz')
        
    model = npzdat['model'].view(np.recarray)
    param = npzdat['param'].view(np.recarray)
    prof = npzdat['prof'].view(np.recarray)

    npzdat.close()

    return model, param, prof


