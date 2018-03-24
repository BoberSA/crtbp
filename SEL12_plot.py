# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:02:06 2018

@author: Stanislav
"""

# COMMON IMPORTS
import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib
#import matplotlib.patches as patches

font = {'family' : 'Times New Roman',
        'size' : 22}
matplotlib.rc('font', **font)
#matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

from crtbp_prop import propCrtbp
from find_vel import findVLimits
from lagrange_pts import lagrange_pts
from stop_funcs import stopFunCombined, iVarX, iVarY, iVarR, iVarR2
#from orbit_geom import orbit_geom
#from jacobi import jacobi_const

recalc = 1

# constants
Sm =  1.9891e30 # mass of the Sun
Em =  5.97e24 # mass of the Earth
ER =  1.496e8 # Sun-Earth distance
mu1 = Sm / (Sm + Em) # CRTBP main coefficient
L = lagrange_pts(mu1) # L1, L2 positions
L1 = L[0, 0]
L2 = L[1, 0]
Ts = 365*24*60*60 / 2 / np.pi


rtol = 1e-12 # integration relative tolerance
nmax = 1e6 # max number of integration steps
int_param = {'atol':rtol, 'rtol':rtol, 'nsteps':nmax}

# planes location
dXkm = 800000
leftpL2 = L2 - dXkm / ER
rightpL2 = L2 + dXkm / ER
topp = 1.0
evL2_L = {'ivar':iVarX, 'stopval':  leftpL2, 'direction': -1, 'isterminal':True, 'corr':False}
evL2_R = {'ivar':iVarX, 'stopval': rightpL2, 'direction':  1, 'isterminal':True, 'corr':False}
evL2_U = {'ivar':iVarY, 'stopval':     topp, 'direction':  0, 'isterminal':True, 'corr':False}
evL2_D = {'ivar':iVarY, 'stopval':    -topp, 'direction':  0, 'isterminal':True, 'corr':False}
evL2 = {'left':[evL2_L], 'right':[evL2_R, evL2_D, evL2_U]}

rightpL1 = L1 + dXkm / ER
leftpL1 = L1 - dXkm / ER
evL1_L = {'ivar':iVarX, 'stopval':  leftpL1, 'direction': -1, 'isterminal':True, 'corr':True}
evL1_R = {'ivar':iVarX, 'stopval': rightpL1, 'direction':  1, 'isterminal':True, 'corr':True}
evL1_U = {'ivar':iVarY, 'stopval':     topp, 'direction':  0, 'isterminal':True, 'corr':False}
evL1_D = {'ivar':iVarY, 'stopval':    -topp, 'direction':  0, 'isterminal':True, 'corr':False}
evL1 = {'left':[evL1_L], 'right':[evL1_R, evL1_D, evL1_U]}

## additional events
evY2 = {'ivar':iVarY, 'stopval':  0, 'direction': 1, 'isterminal':True,  'corr':True}
evY1 = {'ivar':iVarY, 'stopval':  0, 'direction': -1, 'isterminal':True,  'corr':True}

halo_N = 101

L1_halo = np.load('SEL1_halo.npy')
L2_halo = np.load('SEL2_halo.npy')

halo_intrp = scipy.interpolate.interp1d(L1_halo[::-1,2], L1_halo[::-1], \
                                        axis=0, kind='cubic')
L1_halo_i = halo_intrp(np.linspace(L1_halo[0, 2], L1_halo[-1,2], halo_N))

halo_intrp = scipy.interpolate.interp1d(L2_halo[::-1,2], L2_halo[::-1], \
                                        axis=0, kind='cubic')
L2_halo_i = halo_intrp(np.linspace(L2_halo[0, 2], L2_halo[-1,2], halo_N))

if recalc:
    L1_orbs = []
    print('L1')
    for s0 in L1_halo_i:
        print(s0[[0,2]]*ER)
        s0[0] += L1
        v = findVLimits(mu1, s0, 90, evL1, 1e-3, int_param=int_param)
        s0[3:5] += v
        evout = []
        orb=propCrtbp(mu1, s0, [0, 2*np.pi], \
                      stopf=stopFunCombined, events = [evY1], out=evout, \
                      int_param=int_param)
        L1_orbs.append(orb)
        
    L2_orbs = []
    print('L2')
    for s0 in L2_halo_i:
        print(s0[[0,2]]*ER)
        s0[0] += L2
        v = findVLimits(mu1, s0, 90, evL2, 1e-3, int_param=int_param)
        s0[3:5] += v
        evout = []
        orb=propCrtbp(mu1, s0, [0, 2*np.pi], \
                      stopf=stopFunCombined, events = [evY2], out=evout, \
                      int_param=int_param)
        L2_orbs.append(orb)
        
projections = ((0, 1), (0, 2), (1, 2))
axes_names = 'XYZ'

fig, ax = plt.subplots(2, 2, figsize=(20,20), sharex='col', sharey='row')
for p in projections:
    a = ax[p[1]%2, p[0]]
    for orb in L1_orbs:
        a.plot(orb[:,p[0]]*ER,orb[:,p[1]]*ER, '-r', alpha=0.5)
    for orb in L1_orbs[::10]:
        a.plot(orb[:,p[0]]*ER,orb[:,p[1]]*ER, '-k')
    a.set_xlabel(axes_names[p[0]]+', km')
    a.set_ylabel(axes_names[p[1]]+', km')
    a.plot(L[0, p[0]]*ER, L[0, p[1]]*ER, 'ok')
    a.text(L[0, p[0]]*ER, L[0, p[1]]*ER, ' L1')
    a.set_aspect('equal')#'datalim')
fig.delaxes(ax[1, 1])
ax3d = fig.add_subplot(224, projection='3d')
for orb in L1_orbs:
    ax3d.plot(orb[:,0]*ER,orb[:,1]*ER,orb[:,2]*ER, '-r', alpha=0.5)
for orb in L1_orbs[::10]:
    ax3d.plot(orb[:,0]*ER,orb[:,1]*ER,orb[:,2]*ER, '-k')
#ax3d.view_init(elev=10., azim=0.)
ax3d.set_xlabel(axes_names[0]+', km')
ax3d.set_ylabel(axes_names[1]+', km')
ax3d.set_zlabel(axes_names[2]+', km')
fig.tight_layout()
fig.savefig('SEL1_halo_family.png')

fig, ax = plt.subplots(2, 2, figsize=(20,20), sharex='col', sharey='row')
for p in projections:
    a = ax[p[1]%2, p[0]]
    for orb in L2_orbs:
        a.plot(orb[:,p[0]]*ER,orb[:,p[1]]*ER, '-r', alpha=0.5)
    for orb in L2_orbs[::10]:
        a.plot(orb[:,p[0]]*ER,orb[:,p[1]]*ER, '-k')    
    a.set_xlabel(axes_names[p[0]]+', km')
    a.set_ylabel(axes_names[p[1]]+', km')
    a.plot(L[1, p[0]]*ER, L[1, p[1]]*ER, 'ok')
    a.text(L[1, p[0]]*ER, L[1, p[1]]*ER, ' L2')
    a.set_aspect('equal')#'datalim')
fig.delaxes(ax[1, 1])
ax3d = fig.add_subplot(224, projection='3d')
for orb in L2_orbs:
    ax3d.plot(orb[:,0]*ER,orb[:,1]*ER,orb[:,2]*ER, '-r', alpha=0.5)
for orb in L2_orbs[::10]:
    ax3d.plot(orb[:,0]*ER,orb[:,1]*ER,orb[:,2]*ER, '-k')
ax3d.set_xlabel(axes_names[0]+', km')
ax3d.set_ylabel(axes_names[1]+', km')
ax3d.set_zlabel(axes_names[2]+', km')
fig.tight_layout()
fig.savefig('SEL2_halo_family.png')
