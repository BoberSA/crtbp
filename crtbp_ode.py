# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:50:37 2017

@author: Stanislav Bober
"""

import numpy as np
from numba import jit, njit #guvectorize
from numba import compiler, types
from numba.targets import registry

def _crtbp(t, s, mu):
    ''' Right part of Circular Restricted Three Body Problem ODE
        Dimensionless formulation.
        See Murray, Dermott 'Solar System Dynamics'.
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    mu : scalar
         mu = mu1 = m1 / (m1 + m2), 
         where m1 and m2 - masses of two main bodies, m1 > m2
         
    Returns
    -------
    
    ds : np.array
        First order derivative with respect to time of spacecraft
        state vector (vx,vy,vz,dvx,dvy,dvz)
    '''
    
    x, y, z, x1, y1, z1 = s
    mu2 = 1 - mu
    
    yz2 = y * y + z * z;
    r13 = ((x + mu2) * (x + mu2) + yz2) ** 1.5;
    r23 = ((x - mu ) * (x - mu ) + yz2) ** 1.5;

    yzcmn = (mu / r13 + mu2 / r23);

    dx1dt =  2 * y1 + x - (mu * (x + mu2) / r13 + mu2 * (x - mu) / r23);
    dy1dt = -2 * x1 + y - yzcmn * y;
    dz1dt =             - yzcmn * z;
    
    ds = np.array([x1, y1, z1, dx1dt, dy1dt, dz1dt])
    
    return ds

crtbp = compiler.compile_isolated(_crtbp, [types.double, types.double[:], types.double], return_type=types.double[:]).entry_point
#crtbp = registry.CPUDispatcher(_crtbp)
#crtbp.add_overload(crtbp_cr)
