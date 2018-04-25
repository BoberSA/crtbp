# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:57:24 2018

@author: Stanislav
"""

# COMMON IMPORTS
import numpy as np

from mp_pool import pool_run
import os

from lagrange_pts import lagrange_pts
from stop_funcs import iVarX, iVarY, iVarAlpha2
from orbit_geom import make_revs

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
leftp = L1 - 2000000 / ER
topp = 1.0

# events
evR1 =  {'ivar':iVarAlpha2, 'stopval': np.radians(100), \
         'kwargs':{'center':mu1 - 10000 / ER}, \
         'direction': 0, 'isterminal':True, 'corr':False}
evR2 =  {'ivar':iVarAlpha2, 'stopval': np.radians(-100), \
         'kwargs':{'center':mu1 - 10000 / ER}, \
         'direction': 0, 'isterminal':True, 'corr':False}
evL =  {'ivar':iVarX, 'stopval':leftp, 'direction': 0, 'isterminal':True, 'corr':False}
evTm = {'ivar':iVarY, 'stopval': -topp, 'direction': 0, 'isterminal':True, 'corr':False}
evTp = {'ivar':iVarY, 'stopval':  topp, 'direction': 0, 'isterminal':True, 'corr':False}
evStop = {'ivar':iVarY, 'stopval':  0, 'direction': 0, 'isterminal':True, 'corr':False}
events = {'right':[evR1, evR2], 'left':[evL, evTm, evTp], 'stop':evStop}



fmt = '%03d%03d'
nrevs = 5*2 # number of revs
p = 8 
N = 5 # map size
fname = 'mapSEL1_%dx%d_r%d_mp_test_02.h5'%(N,N,nrevs//2)   

def orbit_calc(arg):
    i, j, x0, z0 = arg
    s0 = np.array([L1+x0/ER, 0., z0/ER, 0., 0., 0.])
#    if (i == 0) and (j == 0):
#        print('ZERO.')
#        sys.stdout.flush()
    ret = make_revs(mu1, s0, events, nrevs=nrevs, dv0=(0.05, rtol),
                    maxit=100, retarr=True, prnt=False, int_param=int_param)
#    if (i == 0) and (j == 0):
#        print('ZERO', ret)
    print((fmt+',x=%07d,z=%07d')%(i, j, x0, z0), '+' if ret is not None else '-')
    sys.stdout.flush()
    return ret

def hdf_write(arg):
    i, j, _, _ = arg['job']
    arr = arg['result']
#    if (arg['id'] == 0):
#        print('HDF ZERO')
#    print(('hdf got:'+fmt+' %s')%(i,j, 'None' if arr is None else 'Good'))
    if arr is not None:
        with h5py.File(fname, 'a') as hf:
            hf.create_dataset(fmt%(i,j), data=arr)
            print((fmt+' <write done>')%(i,j))
    
if __name__ == '__main__':

    x  = np.linspace(0.0, 1000000.0, N+1)
    z =  np.linspace(0.0, 1000000.0, N+1)
    zi = np.arange(z.shape[0])
    xi = np.arange(x.shape[0])
    X, Z = np.meshgrid(x, z)
    X = X.reshape(-1,1)
    Z = Z.reshape(-1,1)    
    Xi, Zi = np.meshgrid(xi, zi)
    Xi = Xi.reshape(-1,1)
    Zi = Zi.reshape(-1,1)
    
#    Jobs = np.hstack((Xi.reshape(-1,1), Zi.reshape(-1,1), X.reshape(-1,1), Z.reshape(-1, 1)))    
    Jobs = [(Xi[i,0], Zi[i,0], X[i,0], Z[i,0]) for i in range(len(Xi))]
    
    
    print('Calculation process started in %d processes for %d jobs'%(p, len(Jobs)))
    sys.stdout.flush()
    
    if os.path.isfile(fname):
        os.remove(fname)
    
    pool_run(p, Jobs, orbit_calc, hdf_write)
    
    print('Calculation process finished')
    sys.stdout.flush()
    