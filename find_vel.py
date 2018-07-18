# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:16:34 2017

@author: Stanislav Bober, Guskova Maria
"""

import math
import numpy as np
from crtbp_prop import propCrtbp, prop2Limits, prop2Planes, prop2Spheres
from stop_funcs import stopFunCombined
import scipy.optimize

def findVLimits(mu, y0, beta, lims, dv0, retit=False, maxit=100, **kwargs):
    ''' Calculate velocity correction vector in XY plane that corresponds to \
        bounded motion around libration point in CRTBP.
        Uses modified bisection algorithm; prop2Limits.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    y0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,vx0,vy0).
        
    beta : scalar
        Angle at which correction value will be found.
        
    lims : 
        See prop2Limits function.
            
    dv0 : scalar
        Initial step for correction value calculation.

        
    Optional
    --------
    
    **kwargs : dict
        Parameters for prop2Planes function.
        
    Returns
    -------
    
    v : np.array
      Array of (2,) shape - velocity correction vector
      in XY plane (dvx,dvy)
    
    See Also
    --------
    
    prop2Limits
       
    '''
    y1 = np.asarray(y0).copy()
    vstart = y1[3:5].copy()
    dv = dv0
    dvtol = kwargs.get('dvtol', 1e-16)
    
    rads = math.radians(beta)
    beta_n = np.array([math.cos(rads), math.sin(rads)])
       
    p, _ = prop2Limits(mu, y1, lims, **kwargs)
    y1[3:5] = vstart + dv * beta_n
    p1, _ = prop2Limits(mu, y1, lims, **kwargs)
    
    if p == p1 and p == 1:
        dv = -dv
       
    v = dv        
    i = 0
    while math.fabs(dv) > dvtol and i < maxit:
        y1[3:5] = vstart + v * beta_n
        p1, _ = prop2Limits(mu, y1, lims, **kwargs)
     
        if p1 != p:
            v -= dv
            dv *= 0.5

        v += dv
        i += 1
#    print('findv iterations:', i)
#    print('%g'%v, end=' ')
    if retit:
        return v * beta_n, i
    return v * beta_n


def time2boundary(mu1, s0, v, beta, bnd_events, **kwargs):
    evout = []
    s1 = s0.copy()
    b = math.radians(beta)
    s1[3] += v*math.cos(b)
    s1[4] += v*math.sin(b)
    if type(bnd_events) != list:
        bnd_events = [bnd_events]
    arr = propCrtbp(mu1, s1, [0, 1000], stopf=stopFunCombined, \
              events=bnd_events, out=evout, **kwargs)
    if len(evout) < 1:
        plt.figure(figsize=(15,10))
        plt.plot(arr[:,0], arr[:,1])
        raise RuntimeError("Can't reach boundary")
    return evout[-1][2][6]

def findVmaxT(mu, y0, beta, dv, **kwargs):
    ''' Calculate velocity correction vector in XY plane that corresponds to \
        bounded motion around libration point in CRTBP.
        Uses brent optimization of time to reach specified boundary (i.e. sphere).  

    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    y0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,vx0,vy0).
        
    beta : scalar
        Angle at which correction value will be found.

    dv : scalar
        Defines correction value range: [-dv, dv].
        
    bnd_events : event or list of events
        Event(s) that defines boundary (i.e. sphere)

    Example
    -------
    # Event that describes sphere with center in L2 and 1.5e6 km radius.
    
    _evshpere =  {'ivar':iVarR,
                  'dvar':iVarDR,
                  'stopval':1500000/ER,
                  'direction': 1,
                  'isterminal':True,
                  'corr':True, 
                  'kwargs':{'mu':mu1, 'center':np.array([L2, 0., 0.])}}
    '''

    f = lambda v: -time2boundary(mu, y0, v, beta, **kwargs)
    v_max = scipy.optimize.brent(f, brack=(-dv, dv), tol=1e-16)
    b = math.radians(beta)
    return np.array([v_max*math.cos(b), v_max*math.sin(b)])
    
import matplotlib.pyplot as plt
#from datetime import datetime

def findVLimits_debug(mu, y0, beta, lims, dv0, retit=False, maxit=100, **kwargs):
    ''' Calculate velocity correction vector in XY plane that corresponds to \
        bounded motion around libration point in CRTBP.
        Uses modified bisection algorithm; prop2Limits.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    y0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,vx0,vy0).
        
    beta : scalar
        Angle at which correction value will be found.
        
    lims : 
        See prop2Limits function.
            
    dv0 : scalar
        Initial step for correction value calculation.

        
    Optional
    --------
    
    **kwargs : dict
        Parameters for prop2Planes function.
        
    Returns
    -------
    
    v : np.array
      Array of (2,) shape - velocity correction vector
      in XY plane (dvx,dvy)
    
    See Also
    --------
    
    prop2Limits
       
    '''
    
    lst = []
    y1 = np.asarray(y0).copy()
    vstart = y1[3:5].copy()
    dv = dv0
    dvtol = kwargs.get('dvtol', 1e-16)
    
    rads = math.radians(beta)
    beta_n = np.array([math.cos(rads), math.sin(rads)])
    print(1)
    p, _ = prop2Limits(mu, y1, lims, **kwargs)
    print(2)
    y1[3:5] = vstart + dv * beta_n
    p1, arr = prop2Limits(mu, y1, lims, **kwargs)
    print(3)  
    
    if p == p1 and p == 1:
        dv = -dv
       
    v = dv  
    lst.append(v)
    
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches((15,5))
    
    if p1 == 0:
        ax[0].plot(arr[:,0],arr[:,1], 'b')
        ax[1].plot(arr[:,0],arr[:,2], 'b')
        ax[2].plot(arr[:,1],arr[:,2], 'b')
    else:
        ax[0].plot(arr[:,0],arr[:,1], 'r')
        ax[1].plot(arr[:,0],arr[:,2], 'r')
        ax[2].plot(arr[:,1],arr[:,2], 'r')
    
    i = 0
    while math.fabs(dv) > dvtol and i < maxit:
#        print(v, p, p1)
        y1[3:5] = vstart + v * beta_n
        p1, arr = prop2Limits(mu, y1, lims, **kwargs)
        
        
        if p1 == 0:
            ax[0].plot(arr[:,0],arr[:,1], 'b')
            ax[1].plot(arr[:,0],arr[:,2], 'b')
            ax[2].plot(arr[:,1],arr[:,2], 'b')
        else:
            ax[0].plot(arr[:,0],arr[:,1], 'r')
            ax[1].plot(arr[:,0],arr[:,2], 'r')
            ax[2].plot(arr[:,1],arr[:,2], 'r')
        
        if p1 != p:
            v -= dv
            dv *= 0.5

        v += dv
        lst.append(v)
        i += 1
    y1[3:5] = vstart + v * beta_n
    
    ax[0].set(xlabel='X', ylabel='Y')
    ax[1].set(xlabel='X', ylabel='Z')
    ax[2].set(xlabel='Y', ylabel='Z')
    
    plt.title(str([y1[0], y1[2], y1[4]]))
    fig.tight_layout()
#    plt.savefig('pics/debug '+datetime.now().isoformat().replace(':','-')+'.png')
    print('findv iterations:', i)    
#    print('%g'%v, end=' ')
    if retit:
        return v * beta_n, i
    return v * beta_n

def findVPlanes(mu, s0, beta, planes, dv0, **kwargs):
    ''' Calculate velocity correction vector in XY plane that corresponds to \
        bounded motion around libration point in CRTBP.
        Uses modified bisection algorithm; prop2Planes.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    beta : scalar
        Angle at which correction value will be found.
        
    planes : array_like with 3 components
        See prop2Planes function.
            
    dv0 : scalar
        Initial step for correction value calculation.

        
    Optional
    --------
    
    **kwargs : dict
        Parameters for prop2Planes function.
        
    Returns
    -------
    
    v : np.array
      Array of (2,) shape - velocity correction vector
      in XY plane (dvx,dvy)
    
    See Also
    --------
    
    prop2Planes
       
    '''
    s1 = np.asarray(s0).copy()
    vstart = s1[3:5].copy()
    dv = dv0
    dvtol = kwargs.get('dvtol', 1e-16)
    
    
    rads = math.radians(beta)
    beta_n = np.array([math.cos(rads), math.sin(rads)])
       
    p = prop2Planes(mu, s1, planes, **kwargs)
    s1[3:5] = vstart + dv * beta_n
    p1 = prop2Planes(mu, s1, planes, **kwargs)
    
    if p == p1 and p == 1:
        dv = -dv
       
    v = dv        


    while math.fabs(dv) > dvtol:
        s1[3:5] = vstart + v * beta_n
        p1 = prop2Planes(mu, s1, planes, **kwargs)
     
        if p1 != p:
            v -= dv
            dv *= 0.5

        v += dv
        
#    print('%g'%v, end=' ')
    return v * beta_n

def findVSpheres(mu, s0, beta, spheres, dv0, **kwargs):
    ''' Calculate velocity correction vector in XY plane that corresponds to \
        bounded motion around libration point in CRTBP.
        Uses modified bisection algorithm; prop2Spheres.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    beta : scalar
        Angle at which correction value will be found.
        
    spheres : array_like with 2 components
        See prop2Spheres function.
            
    dv0 : scalar
        Initial step for correction value calculation.

        
    Optional
    --------
    
    **kwargs : dict
        Parameters for prop2Planes function.
        
    Returns
    -------
    
    v : np.array
      Array of (2,) shape - velocity correction vector
      in XY plane (dvx,dvy)
    
    See Also
    --------
    
    prop2Spheres
       
    '''
    s1 = np.asarray(s0).copy()
    vstart = s1[3:5].copy()
    dv = dv0
    dvtol = kwargs.get('dvtol', 1e-16)
    
    
    rads = math.radians(beta)
    beta_n = np.array([math.cos(rads), math.sin(rads)])
    
    p = prop2Spheres(mu, s1, spheres, **kwargs)
    
    if p == 1 and p == 1:
        dv = -dv
        
    v = dv

    while math.fabs(dv) > dvtol:
        s1[3:5] = vstart + v * beta_n
        p1 = prop2Spheres(mu, s1, spheres, **kwargs)
     
        if p1 != p:
            v -= dv
            dv *= 0.5

        v += dv
        
    return v * beta_n