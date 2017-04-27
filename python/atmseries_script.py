import os
#os.chdir('/home/apiso/repos/core_accretion/python')
os.chdir('/Users/ana-mariapiso/core_accretion/python')

import numpy
from RadSGPoly.atmseries_poly import atmseries
from utils.constants import Me

a = [5.0]

for i in range(len(a)):
    atmseries(a[i], 3.2, 2./7, 0.3, 1.0*Me, 1.0*Me, 1, nMpoints = 200,  minMfrac = 1.05, maxMfrac = 2.3)

#for i in range(len(a)):
#    atmseries(a[i], 3.2, 2./7, 0.0, 16.0*Me, 20.0*Me, 5, minMfrac = 1.2, maxMfrac = 2.5)

#for i in range(len(a)):
#    atmseries(a[i], 3.2, 2./5, 0.3, 15.0*Me, 19.0*Me, 5, minMfrac = 1.05, maxMfrac = 2.5)
