from utils.constants import G, kb, mp, Rb, Me, Re, Msun, RH, RHe, sigma, \
     cmperau, RHill, gammafn, mufn, Rfn, Cvfn, kdust, Tdisk, Pdisk, params, yr, RB
from utils.parameters import FT, FSigma, mstar, Y, delad, rhoc, Mc, rc, \
     gamma, Y
import numpy as np
import sys
from numpy import pi
import scipy
import math
import matplotlib.pyplot as plt
from scipy.special import lambertw

def q(a, mc, t = 1e9*yr):
    
    """
    q = RB/Rrcb as limited by the planet's lifetime, when RB < RHill;
    a = semimajor axis in AU
    mc = planet mass in Earth masses
    """
    
    if RB(mc*Me, a) <= RHill(mc*Me, a):
        return np.real(-0.5 * lambertw(-1.09857e9 * a**(28./9) * mc**2 / t**2, 1))
    else:
        return np.real(-0.790106 * a**(4./7) * lambertw(-1.74554e-24 * a**(13./7) * mc**(4./3), 1)) / mc**(2./3)
    
    
a = np.logspace(-1, 2, 100)
mc = np.linspace(1, 10, 100)

qarray = np.ndarray(shape = (len(a), len(mc)), dtype = float)
RBarr = np.ndarray(shape = (len(a), len(mc)), dtype = float)
RHarr = np.ndarray(shape = (len(a), len(mc)), dtype = float)

for i in range(len(a)):
    for j in range(len(mc)):
        qarray[i, j] = q(a[i], mc[j])
        RBarr[i, j] = RB(mc[j]*Me, a[i])
        RHarr[i, j] = RHill(mc[j]*Me, a[i])
        
mm, aa = np.meshgrid(mc, a)

fig = plt.figure(figsize = (5,3.5))
ax = fig.add_subplot(111)
ext = [a[0], a[-1], mc[0], mc[-1]]
ax.set_xscale('log')
im = plt.contourf(aa, mm, qarray, 500)#ax.imshow(qarray, extent = ext)
cbar = fig.colorbar(im)
cbar.set_label(r'$R_{\rm out}/R_{\rm RCB}$')
#ax.set_aspect('auto')
ax.set_xlabel('a [AU]')
ax.set_ylabel(r'$M_{\rm c}$ [$M_{\oplus}$]')
cont = ax.contour(aa, mm, RHarr - RBarr, colors = 'k', levels = [0])
#plt.clabel(cont, inline = 1, fmt='%1.1f')

#plt.draw()
plt.tight_layout()

plt.savefig('../figs/Rout_over_Rrcb_contour.pdf')



