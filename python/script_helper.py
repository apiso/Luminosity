reload(prs)
reload(lsg)
reload(psg)
prms=psg.prms

reload(evap)

filename='a1Mc2'
test=evap.mass_loss(filename, prms, n=17)

prof2, param2, time, Ecool, Eevap, i, flag=test
flag
Ecool, Eevap
param2.rcb[:i+1]
param2.Mcb[:i+1]
time[-1]/yr/1e6