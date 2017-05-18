reload(prs)
reload(lsg)
reload(psg)
prms=psg.prms

reload(evap)

filename='a01Mc2'
test=evap.mass_loss(filename, prms, n=100)

prof2, param2, time, Ecool, Eevap, i, flag, time2=test
flag
Ecool, Eevap
param2.rcb[:i+1]
param2.Mcb[:i+1]
time[-1]/yr/1e6
time2[-1]/yr

plt.semilogy(param2.rcb[1:i+1], time/yr)
plt.semilogy(param2.rcb[1:i+1], time2/yr)