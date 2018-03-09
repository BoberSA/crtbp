# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:45:47 2017

@author: Stanislav
"""

'''
    Researching event detection errors
'''

# COMMON IMPORTS
import numpy as np
import matplotlib.pyplot as plt


from crtbp_prop import propCrtbp
from find_vel import findVPlanes
from lagrange_pts import lagrange1, lagrange2
from stop_funcs import stopFunCombined, iVarX, iVarY, iVarAlpha

# constants
Sm =  1.9891e30 # mass of the Sun
Em =  5.97e24 # mass of the Earth
ER =  1.496e8 # Sun-Earth distance
mu1 = Sm / (Sm + Em) # CRTBP main coefficient
L = [lagrange1(mu1), lagrange2(mu1)] # L1, L2 positions

rtol = 1e-12 # integration relative tolerance
nmax = 1e6 # max number of integration steps
int_param = {'atol':rtol, 'rtol':rtol, 'nsteps':nmax}

# planes location
leftp = mu1 + 500000 / ER
rightp = L[1] + 500000 / ER
topp = 1.0
planes = [leftp, rightp, topp]

# 'small' halo orbit (see preprint https://yadi.sk/d/NaP0529tjsQdJ)
X0km = -277549
Z0km =  200000
# 'big' halo orbit
#X0km = -453098
#Z0km =  500000

# initial spacecraft state vector
y0 = np.array([L[1] + X0km/ER, 0, Z0km/ER, 0, 0, 0])

# find initial velocity for halo orbit
v = findVPlanes(mu1, y0, 90, planes, 0.1, int_param=int_param)
y0[3:5] = v

# events
ev_names = ['X:0', 'alpha:120', 'Y:0', 'alpha:60']
eventX = {'ivar':iVarX, 'stopval':L[1], 'direction': 0, 'isterminal':False, 'corr':True}
eventY = {'ivar':iVarY, 'stopval':   0, 'direction':1, 'isterminal':True,  'corr':True}
eventA = {'ivar':iVarAlpha, 'stopval': np.deg2rad(120), 'direction':0, 'isterminal':False, 'corr':True, 'kwargs':{'center':L[1]}}
eventB = {'ivar':iVarAlpha, 'stopval': np.deg2rad(60), 'direction':0, 'isterminal':False, 'corr':True, 'kwargs':{'center':L[1]}}
evout = []

# integrate CRTBP equations of motion with event detection routine
arr = propCrtbp(mu1, y0, [0, np.pi], stopf=stopFunCombined, events = [eventX, eventA, eventY, eventB], out=evout, int_param=int_param)

# plot orbit projection in XY plane
plt.plot(arr[:,0],arr[:,1],'.-')
plt.axis('equal')

# plot events
for ie, s in evout:
    plt.plot(s[0], s[1], '+k')
    plt.text(s[0], s[1], ' [%d] %s' % (ie, ev_names[ie]))    
