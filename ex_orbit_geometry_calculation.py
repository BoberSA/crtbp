# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:45:47 2017

@author: Stanislav
"""

# COMMON IMPORTS
import numpy as np
import matplotlib.pyplot as plt


#from crtbp_prop import propCrtbp
#from find_vel import findVLimits
from lagrange_pts import lagrange_pts
from stop_funcs import stopFunCombined, iVarX, iVarY, iVarR, iVarDR, iVarAlpha
from orbit_geom import orbit_geom, orbit_geom_old

# constants
Sm =  1.9891e30 # mass of the Sun
Em =  5.97e24 # mass of the Earth
ER =  1.496e8 # Sun-Earth distance
mu1 = Sm / (Sm + Em) # CRTBP main coefficient
L = lagrange_pts(mu1) # L1, L2 positions
L1 = L[0, 0]
L2 = L[1, 0]

rtol = 1e-12 # integration relative tolerance
nmax = 1e6 # max number of integration steps
int_param = {'atol':rtol, 'rtol':rtol, 'nsteps':nmax, 'method':'dop853'}

# planes location
leftp = mu1 + 500000 / ER
rightp = L2 + 500000 / ER
topp = 1.0
planes = [leftp, rightp, topp]

# 'small' halo orbit (see preprint https://yadi.sk/d/NaP0529tjsQdJ)
#X0km = -277549
X0km = -200000
#X0km = -200000
Z0km =  200000
# 'big' halo orbit
#X0km = -453098
#Z0km =  500000

# initial spacecraft state vector
y0 = np.array([L2 + X0km/ER, 0, Z0km/ER, 0, 0, 0])

# events
evL =  {'ivar':iVarX, 'stopval': leftp, 'direction': 0, 'isterminal':True, 'corr':False}
evR =  {'ivar':iVarX, 'stopval':rightp, 'direction': 0, 'isterminal':True, 'corr':False}
evTm = {'ivar':iVarY, 'stopval': -topp, 'direction': 0, 'isterminal':True, 'corr':False}
evTp = {'ivar':iVarY, 'stopval':  topp, 'direction': 0, 'isterminal':True, 'corr':False}
evStop = {'ivar':iVarY, 'stopval': 0, 'direction': -1, 'isterminal':True, 'corr':False}
events = {'left':[evL], 'right':[evR, evTm, evTp], 'stop':evStop}

# findVmaxT test
#evRR  =  {'ivar':iVarR, 'dvar':iVarDR, 'stopval':1400000/ER, 
#          'direction': 0, 'isterminal':True, 'corr':True, 
#          'kwargs':{'mu':mu1, 'center':np.array([L2, 0., 0.])}}
#events = {'bnd_events':[evRR], 'stop':evStop}

#v = findVLimits(mu1, y0, 90, events, 0.1, int_param = int_param)

recalc = [1]
# calc orbit geometry
if recalc[0]:
    lims, arr, dv, evout = orbit_geom(mu1, y0, events, L2, #dv0=(-0.05, 0.05),
                                      dv0=(0.05, rtol), \
                                      nrevs=50, retarr=True, retdv=True, \
#                                      stopf=stopFunCombined, \
                                      retevout=True, int_param=int_param)

# plot orbit projection in XY plane
plt.gcf().set_size_inches((10,10))
plt.plot(arr[:,0],arr[:,1],'.-',markersize=1)#, linewidth=1)
plt.plot(L2, 0, 'ok')
plt.text(L2, 0, '  L2')
for y in lims[1]:
    plt.plot([lims[0][0]+L2, lims[0][1]+L2], [y, y], '--k', linewidth=1)
for x in lims[0]:
    plt.plot([x+L2, x+L2], [lims[1][0], lims[1][1]], '--k', linewidth=1)

for e in evout:
    if e[0] in [0, 1]:
        plt.plot(e[1], e[2], '+k')
    else:
        plt.plot(e[1], e[2], 'og', markersize=4)
        
        
plt.figure(figsize=(10,10))
plt.plot(arr[:,1],arr[:,2],'.-',markersize=1)#, linewidth=1)
plt.plot(0, 0, 'ok')
plt.text(0, 0, '  L2')
#for y in lims[1]:
#    plt.plot([lims[0][0], lims[0][1]], [y, y], '--k', linewidth=1)
#for x in lims[0]:
#    plt.plot([x, x], [lims[1][0], lims[1][1]], '--k', linewidth=1)

for e in evout:
    if e[0] in [0, 1]:
        plt.plot(e[2], e[3], '+k', markersize=8)
    else:
        plt.plot(e[2], e[3], 'og', markersize=4)