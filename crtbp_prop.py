# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:58:00 2017

@author: Stanislav Bober
"""

import numpy as np
import scipy, math
from crtbp_ode import crtbp
from stop_funcs import stopNull, stopFunCombined

def propCrtbp(mu, s0, tspan, **kwargs):
    ''' Propagate spacecraft in CRTBP.
        Uses scipy.integrate.ode with 'dopri5' integrator.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    tspan : array_like with 2 components
        Initial and end time.
        
    Optional
    --------
    
    stopf : function
        Solout function for integrator.
        
    int_param : dict
        Parameters for 'dopri5' integrator.
        
    Returns
    -------
    
    Yt : np.array
      Array of (n,6) shape of state vectors and times
      for each integrator step (xi,yi,zi,vxi,vyi,vzi,ti)
    
    See Also
    --------
    
    scipy.integrate.ode, crtbp_ode.crtbp
       
    '''
    prop = scipy.integrate.ode(crtbp)
    prop.set_initial_value(s0, tspan[0])
    prop.set_f_params(*[mu])
    if 'int_param' in kwargs:
        prop.set_integrator('dopri5', **kwargs['int_param'])
    else:
        prop.set_integrator('dopri5')
    lst = []
    kwargs['mu'] = mu
    #print(kwargs)
    if 'stopf' in kwargs:
        prop.set_solout(lambda t, s: kwargs['stopf'](t, s, lst, **kwargs))
    else:
        prop.set_solout(lambda t, s: stopNull(t, s, lst))
    prop.integrate(tspan[1])
    return np.asarray(lst)

def prop2Limits(mu1, y0, lims, **kwargs):
    """
    lims is a dictionary with terminal event functions
    lims['left'] is a list of events that implement left limit
    lims['right'] is a list of events that implement right limit
    THis function is a copy from planar cr3bp
    
    Returns
    -------
    
    0 : if spacecraft crosses left constrain
    1 : otherwise
    
    and calculated orbit
    """
    evout = []
    arr = propCrtbp(mu1, y0, [0, 3140.0], stopf=stopFunCombined,\
                    events = lims['left']+lims['right'], out=evout, int_param=kwargs['int_param'])
    #print(y0,evout)
    if evout[-1][0] < len(lims['left']):
        return 0, arr
    else:
        return 1, arr
        
def prop2Lyapunov(mu1, y0, lims, **kwargs):
    """
    
    
    Returns
    -------
    
    0 : if spacecraft crosses left constrain
    1 : otherwise
    
    and calculated orbit
    """
    evout = []
    arr = propCrtbp(mu1, y0, [0, 3140.0], stopf=stopFunCombined,\
                    events = lims['left']+lims['right'], out=evout, int_param=kwargs['int_param'])
    #print(y0,evout)
    teta = kwargs.get('teta', 120)
    teta = np.radians(teta)
    L = kwargs.get('L', 1.1557338510267852)
    
    if np.arctan(arr[-1,2], arr[-1, 0] - L) > -teta and np.arctan(arr[-1,2], arr[-1, 0] - L) < teta:
        return 1
    else:
        return 0


def prop2Planes(mu, s0, planes, **kwargs):
    ''' Propagate spacecraft in CRTBP up to defined terminal planes.
        Uses scipy.integrate.ode with 'dopri5' integrator.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    planes : array_like with 3 components
        planes[0] defines terminal plane x == planes[0],
        planes[1] defines terminal plane x == planes[1],
        planes[2] defines 2 terminal planes
            |y| == planes[2].
         
    Optional
    --------
    
    **kwargs : dict
        Parameters for 'dopri5' integrator.
        
    Returns
    -------
    
    0 : if spacecraft crosses plane[0]
    1 : otherwise
    
    See Also
    --------
    
    scipy.integrate.ode, crtbp_ode.crtbp
       
    '''

    def _stopPlanes(t, s, planes=planes):
        if ((s[0] < planes[0]) or (s[0] > planes[1]) or (math.fabs(s[1]) > planes[2])):
            return -1
        return 0
    prop = scipy.integrate.ode(crtbp)
    prop.set_initial_value(s0, 0)
    prop.set_f_params(*[mu])
    if 'int_param' in kwargs:
        prop.set_integrator('dopri5', **kwargs['int_param'])
    else:
        prop.set_integrator('dopri5')
    prop.set_solout(lambda t, s:_stopPlanes(t, s, planes))
    s1 = prop.integrate(3140.0)
    if ((s1[0] > planes[1]) or (math.fabs(s1[1]) > planes[2])):
        return 1
    return 0

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
    
    if p == p1:
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
    print(i)
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





def prop2Spheres(mu, s0, spheres, **kwargs):
    ''' Propagate spacecraft in CRTBP up to defined terminal planes.
        Uses scipy.integrate.ode with 'dopri5' integrator.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    spheres : array_like with 2 components
        spheres[0] defines terminal sphere with center in small body, radius = spheres[0],
        spheres[1] defines terminal sphere with center in small body, radius = spheres[1].
         
    Optional
    --------
    
    **kwargs : dict
        Parameters for 'dopri5' integrator.
        
    Returns
    -------
    
    0 : if spacecraft crosses sphere[0]
    1 : if spacecraft crosses sphere[1]
    
    See Also
    --------
    
    scipy.integrate.ode, crtbp_ode.crtbp
       
    '''

    def _stopSpheres(t, s, mu=mu, spheres=spheres):
        ds = s[:3].copy() # take only coordinates 0,1,2
        ds[0] -= mu # subtract small body position
        r = ds[0]**2+ds[1]**2+ds[2]**2 # calculate radius relative to small body
        r0 = spheres[0]**2
        r1 = spheres[1]**2
        if ((r < r0) or (r > r1)):
            return -1
        return 0
    prop = scipy.integrate.ode(crtbp)
    prop.set_initial_value(s0, 0)
    prop.set_f_params(*[mu])
    if 'int_param' in kwargs:
        prop.set_integrator('dopri5', **kwargs['int_param'])
    else:
        prop.set_integrator('dopri5')
    prop.set_solout(lambda t, s:_stopSpheres(t, s, mu, spheres))
    s1 = prop.integrate(3140.0)
    ds = s1[:3].copy()
    ds[0] -= mu
    r = ds[0]**2+ds[1]**2+ds[2]**2
    r0 = spheres[0]**2
    r1 = spheres[1]**2
    if (r > r1):
        return 1
    return 0

from find_vel import findVPlanes, findVSpheres


def propNRevsPlanes(mu, s0, beta, planes, N=10, dT=np.pi, dv0=0.05, retDV=False, **kwargs):
    ''' Propagate spacecraft in CRTBP for N revolutions near libration point.
        Every dT period calculates velocity correction vector at beta angle
        (findVPlanes) for bounded motion. Initial beta = 90 degrees and
        initial velocity correction vector is velocity itself. 
        Uses propCrtbp, findVPlanes.
    
    Parameters
    ----------
    mu : scalar
        mu = mu1 = m1 / (m1 + m2), 
        where m1 and m2 - masses of two main bodies, m1 > m2.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    beta : angle at which corrections will be made.
        
    planes : array_like with 3 components
        planes[0] defines terminal plane x == planes[0],
        planes[1] defines terminal plane x == planes[1],
        planes[2] defines 2 terminal planes
            |y| == planes[2].
            
    N : scalar
        Number of revolutions.

    dT : scalar
        Time between velocity corrections.
        (default: pi, i.e. about one spacecraft revolution)

    dv0 : scalar
        Initial step for correction calculation.

    retDV : boolean
        If True then additionally returns array with all
        correction dV vectors.    
         
    Optional
    --------
    
    **kwargs : dict
        Parameters for propCrtbp and findVPlanes.
        
    Returns
    -------
    
    arr : np.array
      Array of (n,6) shape of state vectors and times
      for each integrator step (xi,yi,zi,vxi,vyi,vzi,ti)
    
    DV : np.array
        Array of (N,) shape with correction values.
        (only if retDV is True)
    
    See Also
    --------
    
    propCrtbp, findVPlanes
       
    '''
    pb = None
    if 'progbar' in kwargs:
        pb = kwargs.pop('progbar')
        
    v = findVPlanes(mu, s0, 90, planes, dv0, **kwargs)
    s1 = s0.copy()
    s1[3:5] += v
    DV = v.copy()
    cur_rev = propCrtbp(mu, s1, [0, dT], **kwargs)
    arr = cur_rev.copy()
    #print(0, end=' ')
    for i in range(N):
        s1 = arr[-1, :-1].copy()
        v = findVPlanes(mu, s1, beta, planes, dv0, **kwargs)
        s1[3:5] += v
        cur_rev = propCrtbp(mu, s1, [0, dT], **kwargs)
        cur_rev[:,-1]+=arr[-1,-1]
        arr = np.vstack((arr, cur_rev[1:]))
        DV = np.vstack((DV, v))
        if pb:
            pb.value=i+1
        else:
            print(i+1, end=' ')
    if retDV:
        return arr, DV
    return arr


def propNRevsSpheres(mu, s0, beta, spheres, N=10, dT=np.pi, dv0=0.05, retDV=False, **kwargs):
    ''' Propagate spacecraft in CRTBP for N revolutions near libration point.
        Every dT period calculates velocity correction vector at beta angle
        (findVPlanes) for bounded motion. Initial beta = 90 degrees and
        initial velocity correction vector is velocity itself. 
        Uses propCrtbp, findVPlanes.
    
    Parameters
    ----------
    mu : scalar
        mu = mu1 = m1 / (m1 + m2), 
        where m1 and m2 - masses of two main bodies, m1 > m2.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
        
    beta : angle at which corrections will be made.
        
    spheres : array_like with 2 components
        spheres[0] defines terminal sphere with center in small body, radius = spheres[0],
        spheres[1] defines terminal sphere with center in small body, radius = spheres[1].
            
    N : scalar
        Number of revolutions.

    dT : scalar
        Time between velocity corrections.
        (default: pi, i.e. about one spacecraft revolution about libration point)

    dv0 : scalar
        Initial step for correction calculation.

    retDV : boolean
        If True then additionally returns array with all
        correction dV vectors.    
         
    Optional
    --------
    
    **kwargs : dict
        Parameters for propCrtbp and findVPlanes.
        
    Returns
    -------
    
    arr : np.array
      Array of (n,6) shape of state vectors and times
      for each integrator step (xi,yi,zi,vxi,vyi,vzi,ti)
    
    DV : np.array
        Array of (N,) shape with correction values.
        (only if retDV is True)
    
    See Also
    --------
    
    propCrtbp, findVPlanes
       
    '''
    pb = None
    if 'progbar' in kwargs:
        pb = kwargs.pop('progbar')
        
    v = findVSpheres(mu, s0, 90, spheres, dv0, **kwargs)
    s1 = s0.copy()
    s1[3:5] += v
    DV = v.copy()
    cur_rev = propCrtbp(mu, s1, [0, dT], **kwargs)
    arr = cur_rev.copy()
    #print(0, end=' ')
    for i in range(N):
        s1 = arr[-1, :-1].copy()
        v = findVSpheres(mu, s1, beta, spheres, dv0, **kwargs)
        s1[3:5] += v
        cur_rev = propCrtbp(mu, s1, [0, dT], **kwargs)
        cur_rev[:,-1]+=arr[-1,-1]
        arr = np.vstack((arr, cur_rev[1:]))
        DV = np.vstack((DV, v))
        if pb:
            pb.value=i+1
        else:
            print(i+1, end=' ')
    if retDV:
        return arr, DV
    return arr