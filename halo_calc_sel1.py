# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:45:43 2018

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

from crtbp_prop import propCrtbp
from find_vel import findVLimits, findVLimits_debug
from lagrange_pts import lagrange_pts
from stop_funcs import stopFunCombined, iVarX, iVarY, iVarR, iVarR2
#from orbit_geom import orbit_geom
#from jacobi import jacobi_const

recalc = 0

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

# additional events
evY = {'ivar':iVarY, 'stopval':  0, 'direction': 1, 'isterminal':True,  'corr':True}


halo_z0 = np.arange(700000, 0, -50000)

def halo_goal(x, z, mu1, **kwargs):
    print(x*ER, end=' ')
    s0 = np.array([L1 + x, 0, z, 0, 0, 0])
    v = findVLimits(mu1, s0, 90, evL1, 0.2, **kwargs)
    s0[3:5] += v
    evout = []
    propCrtbp(mu1, s0, [0, 2*np.pi], \
                stopf=stopFunCombined, events = [evY], out=evout, \
                **kwargs)
    evout.pop(0)
    vx = evout[-1][2][3]
    vz = evout[-1][2][5]
    return vx**2 + vz**2

halo_x0 = []

for z in halo_z0:
    print('\nz = ', z)
    f = lambda x: halo_goal(x, z/ER, mu1, int_param=int_param)
    opt = scipy.optimize.fminbound(f, 0, 1000000/ER, full_output=True, xtol=1e-6)
    halo_x0.append(opt[0]*ER)    