# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:29:19 2018

@author: stasb
"""

# COMMON IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from mp_pool import pool_run
import os

from crtbp_prop import propCrtbp
from lagrange_pts import lagrange_pts
from stop_funcs import iVarX, iVarVX, iVarY, iVarVY, iVarAlpha, stopFunCombined
#from orbit_geom import make_revs

import sys
import warnings

# disable warnings because of stupid h5py warning
# run with -W key to get warnings back
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import h5py

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
# integrator parameters
int_param = {'atol':rtol, 'rtol':rtol, 'nsteps':nmax, 'method':'dop853'}

# planes location
dXkm = 1000000
rightp = L2 + dXkm / ER
leftp = L2 - dXkm / ER
topp = 2*dXkm / ER

# events
#evL =  {'ivar':iVarAlpha, 'stopval': np.radians(135), \
#        'kwargs':{'center':mu1 + 1 / ER}, \
#        'direction': 0, 'isterminal':True, 'corr':False}
evL =  {'ivar':iVarX, 'dvar':iVarVX, 'stopval': leftp, 'direction': 0, 'isterminal':True, 'corr':True}
evR =  {'ivar':iVarX, 'dvar':iVarVX, 'stopval':rightp, 'direction': 0, 'isterminal':True, 'corr':True}
evTm = {'ivar':iVarY, 'dvar':iVarVY, 'stopval': -topp, 'direction': 0, 'isterminal':True, 'corr':True}
evTp = {'ivar':iVarY, 'dvar':iVarVY, 'stopval':  topp, 'direction': 0, 'isterminal':True, 'corr':True}
lims = {'left':[evL], 'right':[evR, evTm, evTp]}


sel2_halo = np.load('SEL2_halo.npy')

fmt = '%03d'
dvp = 30 # minV = V*(1-dvp/100), maxV = V*(1+dvp/100)
idx = 0 # halo number
N = 256 # map size
M = 128 # minber of [minV, maxV] interval divisions

p = 8 # number of processes

fname = 'su_mapSEL2_h%02d_N%02d_dvp%02d.h5'%(idx, N, dvp)  
    
def calc(arg):
    _, s, v, n = arg
    
    A = np.linspace(0, np.pi, n)
    B = np.linspace(0, 2*np.pi, 2*n)
    T = np.zeros((2*n, n, 2))
    
    s0 = s.copy()
    s0[0] += L2
    for i, a in enumerate(A):
#        print('\n', i, end='\n\t')
        print(i, end = ' ')
        for j, b in enumerate(B):
#            print(j, end=' ')
            s0[3] = v*np.sin(a)*np.cos(b)
            s0[4] = v*np.sin(a)*np.sin(b)
            s0[5] = v*np.cos(a)
            for k in range(2):
                evout = []
                propCrtbp(mu1, s0, [0, [1, -1][k]*3140.0], stopf=stopFunCombined,\
                          events=lims['left']+lims['right'], out=evout, int_param=int_param)
#                evout.pop(0)
                if evout[-1][0] < len(lims['left']):
                    T[j, i, k] = -evout[-1][2][6]
                else:
                    T[j, i, k] = evout[-1][2][6]
            
    
#    print('%03d%03d,x=%07d,z=%07d'%(i, j, x0, z0), '+' if ret is not None else '-')
#    sys.stdout.flush()
    return T


def hdf_write(arg):
    i, _, _, _ = arg['job']
    arr = arg['result']

    if arr is not None:
        with h5py.File(fname, 'a') as hf:
            hf.create_dataset(fmt%(i), data=arr)
            print((fmt+' <write done>')%(i))
#    
if __name__ == '__main__':

    s = sel2_halo[0]
    v0 = sel2_halo[0, 4]

    minV = v0*(1-dvp/100.0)
    maxV = v0*(1+dvp/100.0)
    
    V = np.linspace(minV, maxV, M)

    A = np.linspace(0, np.pi, N)
    B = np.linspace(0, 2*np.pi, 2*N)
    
    with h5py.File(fname, 'w') as hf:
        g = hf.create_group('meta')
        g.create_dataset('S0', data=s)
        g.create_dataset('V', data=V)
        g.create_dataset('alpha', data=A)
        g.create_dataset('beta', data=B)

    Jobs = [(i, s, v, N) for i, v in enumerate(V)]

    pool_run(p, Jobs, calc, hdf_write)


#    T = calc((0, s, v0, N))
#    
#
#    
#    AA, BB = np.meshgrid(A, B)
#    
#    plt.figure(figsize=(10,10))
#    plt.contour(AA, BB, T[:,:,0], levels=[0], cmap='winter')
#    plt.contour(AA, BB, T[:,:,1], levels=[0], cmap='autumn')
#    plt.axis('equal')
#    
#    plt.figure(figsize=(5,10))
#    plt.contourf(AA, BB, T[:,:,0], cmap='jet')
#    
#    plt.figure(figsize=(5,10))
#    plt.contourf(AA, BB, T[:,:,1], cmap='jet')
#    
#    plt.figure(figsize=(5,10))
#    
#    s0 = s.copy()
#    s0[0] += L2
#    k = 0
#    i, j = np.unravel_index(np.argmax(T[:,:,k], axis=None), T[:,:,k].shape)
#    a = A[i]
#    b = B[j]
#    s0[3] = v0*np.sin(a)*np.cos(b)
#    s0[4] = v0*np.sin(a)*np.sin(b)
#    s0[5] = v0*np.cos(a)
#    evout = []
#    arr = propCrtbp(mu1, s0, [0, [1, -1][k]*3140.0], stopf=stopFunCombined,\
#                    events=lims['left']+lims['right'], out=evout, int_param=int_param)
#    plt.plot(arr[:,0], arr[:,1])
#    plt.axis('equal')
