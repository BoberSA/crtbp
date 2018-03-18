# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:16:34 2017

@author: Stanislav Bober, Guskova Maria
"""

import math
import numpy as np
from crtbp_prop import prop2Limits, prop2Planes, prop2Spheres

def findVLimits(mu, y0, beta, lims, dv0, **kwargs):
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
    
    if p == p1 and p == 0:
        dv = -dv
       
    v = dv        
    i = 0
    while math.fabs(dv) > dvtol and i < 100:
        y1[3:5] = vstart + v * beta_n
        p1, _ = prop2Limits(mu, y1, lims, **kwargs)
     
        if p1 != p:
            v -= dv
            dv *= 0.5

        v += dv
        i += 1
#    print(i)
#    print('%g'%v, end=' ')
    return v * beta_n


import matplotlib.pyplot as plt
from datetime import datetime

def findVLimits_debug(mu, y0, beta, lims, dv0, **kwargs):
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
       
    p, _ = prop2Limits(mu, y1, lims, **kwargs)
    y1[3:5] = vstart + dv * beta_n
    p1, arr = prop2Limits(mu, y1, lims, **kwargs)
    
    if p == p1:
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
    while math.fabs(dv) > dvtol and i < 150:
        #print(v, p, p1)
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
    
    plt.savefig('pics/debug '+datetime.now().isoformat().replace(':','-')+'.png')
        
#    print('%g'%v, end=' ')
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
    
    if p == p1:
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
    if p == 1:
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