# -*- coding: utf-8 -*-
"""

This module contains the initial parameters in order to create a model
atmosphere:

    Disk parameters: r, FT, FSigma, mstar
    Y: Helium mass fraction
    R: specific gas constant or a given Y
    Mc, rc, rhoc: core parameters
    beta = the temperature dependence factor in the opacity
    
"""

import numpy as np
from constants import Me, Rb, Rfn


#disk parameters (relative to MMSN model)
a = 60 #disk radius in AU
FT = 1.0
FSigma = 1.0
mstar = 1.0 #star mass in solar masses

#adiabatic indices
gamma = 7./5
delad = 1 - 1/gamma

#parameters for H-He mixture
Y = 0.3
muH = 2.0
mu = 4./(2-Y)
R = Rb/mu
Cv = R / (gamma - 1)
R_EOS = Rfn(Y)


#core parameters
Mc = 5*Me
rhoc = 2.0
rc = (3*Mc/(4*np.pi*rhoc))**(1./3)


#opacity
beta = 2


#ncol=15 #14
