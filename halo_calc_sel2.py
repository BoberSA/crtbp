# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:45:43 2018

@author: Stanislav
"""

# COMMON IMPORTS
import numpy as np
import scipy.interpolate
import scipy.optimize

from crtbp_prop import propCrtbp
from find_vel import findVLimits
from lagrange_pts import lagrange_pts
from stop_funcs import stopFunCombined, iVarX, iVarY, iVarVY

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

# additional events
evY = {'ivar':iVarY, 'dvar':iVarVY, 'stopval':  0, 'direction': 1, 'isterminal':True,  'corr':True}

# Az range - z-amplitudes of halo orbits (km)
halo_z0 = np.arange(700000, 0, -50000)

sel2_halo = np.zeros((halo_z0.shape[0], 6))
sel2_halo[:,2] = halo_z0 / ER

def halo_goal(x, z, mu1, retv=False, **kwargs):
    print(x*ER, end=' ')
    s0 = np.array([L2 + x, 0, z, 0, 0, 0])
    v = findVLimits(mu1, s0, 90, evL2, 0.2, **kwargs)
    s0[3:5] += v
    evout = []
    propCrtbp(mu1, s0, [0, 2*np.pi], \
                stopf=stopFunCombined, events = [evY], out=evout, \
                **kwargs)
#    evout.pop(0)
    vx = evout[-1][2][3]
    vz = evout[-1][2][5]
    if retv:
        return vx**2 + vz**2, v
    return vx**2 + vz**2

for i, s0 in enumerate(sel2_halo):
    print('[%02d]'%i, end = ' ')
    z = s0[2]
    f = lambda x: halo_goal(x, z, mu1, int_param=int_param)
    opt = scipy.optimize.fminbound(f, -1000000/ER, 0, full_output=True, xtol=rtol)
    x = opt[0]
    _, v = halo_goal(x, z, mu1, retv=True, int_param=int_param)
    print('\nz =', z, 'x =', x, 'v =', v)
    sel2_halo[i, 0] = x
    sel2_halo[i, 4] = v[1]

# save initial state vectors for halo to file
np.save('SEL2_halo.npy', sel2_halo)

