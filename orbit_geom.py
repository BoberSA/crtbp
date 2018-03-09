# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:20:41 2018

@author: Stanislav
"""

import numpy as np
from crtbp_prop import propCrtbp
from find_vel import findVLimits
#from lagrange_pts import lagrange1, lagrange2
from stop_funcs import stopFunCombined, iVarY, iVarVY, iVarT

def orbit_geom(mu, s0, events, center, beta=(90, 0), nrevs=10, dv0=0.05, \
               retarr=False, retdv=False, retevout=False, T=None, **kwargs):
    ''' Orbit geometry calculation function.
        Propagates orbit for 'nrevs' revolutions using findVLimits function \
        after each revolution for station-keeping.
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.

    s0 : array_like with 6 components
        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).

    events : dict
        events['left'] should be termination event list for left boundary
        events['right'] should be termination event list for left boundary
        Optionally:
        events['stop'] should be termination event (list of events) that
        defines one revolution
        
    center : float
        Coordinate system center for X-axis. Necessary for separation
        orbit geometry along X-axis.

    beta : list of 2 floats
        Angles for initial (beta[0]) and all others (beta[1]) calculation
        of correction burns (delta-v).
        
    nrevs : int
        Number of revolutions.
        
    dv0 : float
        Initial step size for delta-v calculation.
        
    Optional
    --------
    
    T : float
        Defines one revolution period if events['stop'] wasn't defined.
    
    retarr : bool
        If True function returns full orbit array (all integration steps).
        
    retdv : bool
        If True function returns all calculated correction burns (delta-v).
            
    retevout : bool
        If True function returns all calculated geometric events.
              
    Returns
    -------
    
    lims : tuple of 3 tuples of 4 floats
        (min(X|X<0), max(X|X>0), max(X|X<0), min(X|X>0)) and the same 
        for Y and Z coordinates. X is relative to 'center' coordinate.
    
    arr : numpy array of (N, len(s0)+1+2+len(evStop)) shape
        Spacecraft 'extended' state vectors for all (N) integration steps.
        
    dv : numpy array of (nrevs, 2) shape
        All calculated correction burns (delta-v) in XY plane.
        
    evout : numpy array of (M, 1 + len(s0)+1+2+len(evStop)) shape
        Each row consists of: index of event and 'extended' state vector.
        
          
    '''
    evY =  {'ivar': iVarY, 'stopval':   0.0, 'direction': 0, 'isterminal':False, 'corr':True } #0
    evVY = {'ivar':iVarVY, 'stopval':   0.0, 'direction': 0, 'isterminal':False, 'corr':True } #1
    # default "one revolution" terminal event
    evTrm = {'ivar':iVarT, 'stopval': np.pi, 'direction': 0, 'isterminal':True,  'corr':False}
    evLeft = events['left']
    evRight = events['right']
    evStop = events.get('stop', evTrm)
    if type(evLeft) is not list:
        evLeft = [evLeft]
    if type(evRight) is not list:
        evRight = [evRight]
    if type(evStop) is not list:
        evStop = [evStop]

    evGeom = [evY, evVY]
    evout = []
    
    sn = len(s0)
    
    s1 = s0.copy()
    v = findVLimits(mu, s1, beta[0], {'left':evLeft, 'right':evRight}, dv0, **kwargs)
    s1[3:5] += v
    DV = v.copy().reshape(1, -1)
    if (T is None) and (evStop[0]['ivar'] == iVarT):
        T = 2*evStop[0]['stopval']
    else:
        T = 100*np.pi
#    print('T=', T)
    cur_rev = propCrtbp(mu, s1, [0, T], stopf=stopFunCombined, \
                        events = evGeom + evStop, \
                        out=evout, **kwargs)
    arr = cur_rev.copy()
    print(0, end=' ')
    for i in range(nrevs):
        s1 = arr[-1, :sn].copy()
#        dv0 = np.linalg.norm(DV[-1])
        v = findVLimits(mu, s1, beta[1], events, dv0, **kwargs)
        s1[3:5] += v
        cur_rev = propCrtbp(mu, s1, [0, T], stopf=stopFunCombined,
                            events = evGeom + evStop,
                            out=evout, **kwargs)
        cur_rev[:,sn]+=arr[-1,sn]
        arr = np.vstack((arr, cur_rev[1:]))
        DV = np.vstack((DV, v))
        print(i+1, end=' ')
    
    evout.pop(0)
    evout = np.array([[e[0], *e[2]] for e in evout])
    
    evY = evout[(evout[:,0]==1)]
    maskYp = (evY[:,2] >= 0.0)
    Yp = evY[maskYp,2]
    Ym = evY[np.logical_not(maskYp),2]
    evXZ = evout[(evout[:,0]==0)]
    evXZ[:,1] -= center
    maskXp = (evXZ[:,1] >= 0.0)
    Xp = evXZ[maskXp, 1]
    Xm = evXZ[np.logical_not(maskXp), 1]
    maskZp = (evXZ[:,3] >= 0.0)
    Zp = evXZ[maskZp, 3]
    Zm = evXZ[np.logical_not(maskZp), 3]

    ylim = (np.min(Ym), np.max(Yp), np.max(Ym), np.min(Yp))
    xlim = (np.min(Xm), np.max(Xp), np.max(Xm), np.min(Xp))
    zlim = (np.min(Zm), np.max(Zp), np.max(Zm), np.min(Zp))
    
    ret = [(xlim, ylim, zlim)]
    
    if retarr:
        ret.append(arr)
        
    if retdv:
        ret.append(DV)
    
    if retevout:
        ret.append(evout)
        
    if len(ret) == 1:
        return ret[0]
    
    return tuple(ret)


#def propNRevsPlanes(mu, s0, beta, planes, N=10, dT=np.pi, dv0=0.05, retDV=False, **kwargs):
#    ''' Propagate spacecraft in CRTBP for N revolutions near libration point.
#        Every dT period calculates velocity correction vector at beta angle
#        (findVPlanes) for bounded motion. Initial beta = 90 degrees and
#        initial velocity correction vector is velocity itself. 
#        Uses propCrtbp, findVPlanes.
#    
#    Parameters
#    ----------
#    mu : scalar
#        mu = mu1 = m1 / (m1 + m2), 
#        where m1 and m2 - masses of two main bodies, m1 > m2.
#
#    s0 : array_like with 6 components
#        Initial spacecraft state vector (x0,y0,z0,vx0,vy0,vz0).
#        
#    beta : angle at which corrections will be made.
#        
#    planes : array_like with 3 components
#        planes[0] defines terminal plane x == planes[0],
#        planes[1] defines terminal plane x == planes[1],
#        planes[2] defines 2 terminal planes
#            |y| == planes[2].
#            
#    N : scalar
#        Number of revolutions.
#
#    dT : scalar
#        Time between velocity corrections.
#        (default: pi, i.e. about one spacecraft revolution)
#
#    dv0 : scalar
#        Initial step for correction calculation.
#
#    retDV : boolean
#        If True then additionally returns array with all
#        correction dV vectors.    
#         
#    Optional
#    --------
#    
#    **kwargs : dict
#        Parameters for propCrtbp and findVPlanes.
#        
#    Returns
#    -------
#    
#    arr : np.array
#      Array of (n,6) shape of state vectors and times
#      for each integrator step (xi,yi,zi,vxi,vyi,vzi,ti)
#    
#    DV : np.array
#        Array of (N,) shape with correction values.
#        (only if retDV is True)
#    
#    See Also
#    --------
#    
#    propCrtbp, findVPlanes
#       
#    '''
#    pb = None
#    if 'progbar' in kwargs:
#        pb = kwargs.pop('progbar')
#        
#    v = findVPlanes(mu, s0, 90, planes, dv0, **kwargs)
#    s1 = s0.copy()
#    s1[3:5] += v
#    DV = v.copy()
#    cur_rev = propCrtbp(mu, s1, [0, dT], **kwargs)
#    arr = cur_rev.copy()
#    #print(0, end=' ')
#    for i in range(N):
#        s1 = arr[-1, :-1].copy()
#        v = findVPlanes(mu, s1, beta, planes, dv0, **kwargs)
#        s1[3:5] += v
#        cur_rev = propCrtbp(mu, s1, [0, dT], **kwargs)
#        cur_rev[:,-1]+=arr[-1,-1]
#        arr = np.vstack((arr, cur_rev[1:]))
#        DV = np.vstack((DV, v))
#        if pb:
#            pb.value=i+1
#        else:
#            print(i+1, end=' ')
#    if retDV:
#        return arr, DV
#    return arr