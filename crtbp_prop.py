# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:58:00 2017

@author: Stanislav Bober
"""

import numpy as np
import scipy, math
from crtbp_ode import crtbp
from stop_funcs import stopNull

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