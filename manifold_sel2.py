# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:08:09 2018

@author: Stanislav

This code calculates and plots unstable and stable manifolds for halo
orbit near L2 point.

"""

# COMMON IMPORTS
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
from colorline import colorline

font = {'family' : 'Times New Roman', 'size' : 16}
matplotlib.rc('font', **font)
#matplotlib inline

from crtbp_prop import propCrtbp
from find_vel import findVPlanes
from lagrange_pts import lagrange1, lagrange2
from stop_funcs import stopFunCombined, iVarY

# use 0 for plot without recalculation manifolds, 1 - with recalculation
recalc = 0

# constants
Sm =  1.9891e30 # mass of the Sun
Em =  5.97e24 # mass of the Earth
ER =  1.496e8 # Sun-Earth distance
mu1 = Sm / (Sm + Em) # CRTBP main coefficient
L = np.array([[lagrange1(mu1), lagrange2(mu1)], [0.0, 0.0], [0.0, 0.0]]).T # L1, L2 positions

rtol = 1e-12 # integration relative tolerance
nmax = 1e6 # max number of integration steps
int_param = {'atol':rtol, 'rtol':rtol, 'nsteps':nmax}

# planes location
dXkm = 500000
leftpL2 = mu1 + dXkm / ER
rightpL2 = L[1,0] + dXkm / ER
topp = 1.0
planesL2 = [leftpL2, rightpL2, topp]

# 'small' halo orbit (see preprint https://yadi.sk/d/NaP0529tjsQdJ)
X0km = -277549
Z0km =  200000
# 'big' halo orbit
#X0km = -453098
#Z0km =  500000

# number of trajectories for manifolds
Nmanif = 200

# time period to propagate manifolds
T = ( 1.5*np.pi, # UNSTABLE FROM EARTH
      1.5*np.pi, # UNSTABLE TO EARTH
     -1.5*np.pi, # STABLE FROM EARTH
     -1.5*np.pi) # STABLE TO EARTH

# geometric event for integration of initial orbit for one revotulion
eventY = {'ivar':iVarY, 'stopval':   0, 'direction':1, 'isterminal':True,  'corr':True}

# projection definition
projections = ((0, 1), (0, 2), (1, 2))
axes_names = 'XYZ'

# figure titles
titles = ('Unstable manifold', 'Stable manifold')

if recalc:
    # initial spacecraft state vector
    y0 = np.array([L[1,0] + X0km/ER, 0, Z0km/ER, 0, 0, 0])
    print('Calculating initial state velocity')
    # find initial velocity for halo orbit
    v = findVPlanes(mu1, y0, 90, planesL2, 0.1, int_param=int_param)
    y0[3:5] = v
    print('v =', v)


    # integrate CRTBP equations of motion with event detection routine
    evout = []
    arr = propCrtbp(mu1, y0, [0, np.pi], stopf=stopFunCombined, events = [eventY], out=evout, int_param=int_param)


# plot initial orbit
fig, ax = plt.subplots(2, 2, figsize=(10,10), sharex='col', sharey='row')
for p in projections:
    a = ax[p[1]%2, p[0]]
    a.plot(arr[:,p[0]],arr[:,p[1]])
    a.set_xlabel(axes_names[p[0]])
    a.set_ylabel(axes_names[p[1]])
    a.plot(L[1, p[0]], L[1, p[1]], 'ok')
    a.text(L[1, p[0]], L[1, p[1]], ' L2')
    a.set_aspect('equal')#'datalim')
fig.delaxes(ax[1, 1])
ax3d = fig.add_subplot(224, projection='3d')
ax3d.plot(arr[:,0],arr[:,1],arr[:,1])
ax3d.set_xlabel(axes_names[0])
ax3d.set_ylabel(axes_names[1])
ax3d.set_zlabel(axes_names[2])
fig.tight_layout()

if recalc:
    # calculate equally distributed state vectors
#    l = np.cumsum(np.linalg.norm(arr[1:,:3]-arr[:-1,:3], axis=1)) # by length
    l = arr[1:,6] # by time
    intrp = scipy.interpolate.interp1d(l, arr[1:], axis=0, kind='cubic', copy=False)#, assume_sorted=False)
    ls = np.linspace(l[0], l[-1], Nmanif)
    pts0 = intrp(ls)

# copy data for editing (perturbation)
pts = pts0.copy()

if recalc:
    # MANIFOLD calculation
    MANIF = [[], [], [], []]
    print('Calculating', Nmanif, 'manifold trajectories')    
    pts[:,3] += 0.000001 # small perturbation for Vx
    for i in range(len(pts)):
        print(i, end = ' ')
        arr1 = propCrtbp(mu1, pts[i,0:6], [0, T[0]], int_param=int_param)
        MANIF[0].append(arr1)
        arr1 = propCrtbp(mu1, pts[i,0:6], [0, T[2]], int_param=int_param)
        MANIF[2].append(arr1)
    pts[:,3] -= 0.000002
    for i in range(len(pts)):
        print(i, end = ' ')
        arr1 = propCrtbp(mu1, pts[i,0:6], [0, T[1]], int_param=int_param)
        MANIF[1].append(arr1)
        arr1 = propCrtbp(mu1, pts[i,0:6], [0, T[3]], int_param=int_param)
        MANIF[3].append(arr1)

# plot manifolds
print('\nPlotting manifolds')
fig, ax = plt.subplots(4, 2, figsize=(10,20), sharex='col', sharey='row')
for i in range(4):
    for p in projections:
        a = ax[(i//2*2+p[1]%2), p[0]]
        for j, tr in enumerate(MANIF[i]):
            colorline(tr[:,p[0]], tr[:,p[1]], c=tr[:,6], ax=a, cmap='jet', 
                      norm=plt.Normalize(0, T[i]) if T[i] > 0.0 else plt.Normalize(T[i], 0), 
                      alpha=0.5)
        a.set_xlabel(axes_names[p[0]])
        a.set_ylabel(axes_names[p[1]])
        a.set_title(titles[i//2])
        a.plot(L[1, p[0]], L[1, p[1]], 'ok')
        a.text(L[1, p[0]], L[1, p[1]], ' L2')
        a.set_aspect('equal', 'datalim')
    print(' '.join(titles).split()[i], end = ' ')

# calculate min and max values for x, y and z coordinates of manifolds
xlim = [2.0, 0.0]*2
ylim = [2.0, -2.0]*2
zlim = [2.0, -2.0]*2
for i in range(4):
    j = i//2*2
    for tr in MANIF[i]:
        xlim[0+j] = min([xlim[0+j], np.min(tr[:,0])])
        xlim[1+j] = max([xlim[1+j], np.max(tr[:,0])])
        ylim[0+j] = min([ylim[0+j], np.min(tr[:,1])])
        ylim[1+j] = max([ylim[1+j], np.max(tr[:,1])])
        zlim[0+j] = min([zlim[0+j], np.min(tr[:,2])])
        zlim[1+j] = max([zlim[1+j], np.max(tr[:,2])])

# plot 3D manifolds
fig.delaxes(ax[1, 1])
fig.delaxes(ax[3, 1])
ax3d = [None]*2
ax3d[0] = fig.add_subplot(424, projection='3d')
ax3d[1] = fig.add_subplot(428, projection='3d')

for i in range(4):
    a = ax3d[i//2]
    for tr in MANIF[i]:
        j = i//2*2
        colorline(tr[:,0], tr[:,1], tr[:,2], c=tr[:,6], ax=a, cmap='jet', 
                  norm=plt.Normalize(0, T[i]) if T[i] > 0.0 else plt.Normalize(T[i], 0), 
                  alpha=0.5)
    a.set_xlim(xlim[(0+j):(1+j+1)])
    a.set_ylim(ylim[(0+j):(1+j+1)])
    a.set_zlim(zlim[(0+j):(1+j+1)])
    a.set_xlabel(axes_names[0])
    a.set_ylabel(axes_names[1])
    a.set_zlabel(axes_names[2])
    a.set_title(titles[i//2])

fig.tight_layout()
print('\nShowing figure...')