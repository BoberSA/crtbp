# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:55:16 2017

@author: Stanislav Bober
"""

import numpy as np
import math
from scipy.interpolate import interp1d

'''
    NEW FUNCTIONS 16-07-2017
'''

def iVarX(t, s, **kwargs):
    ''' Independent variable function that returns X coordinate \
    of spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    X coordinate of state vector s.
    '''
    return s[0]
    

def iVarY(t, s, **kwargs):
    ''' Independent variable function that returns Y coordinate \
    of spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    Y coordinate of state vector s.
    '''    
    return s[1]

def iVarZ(t, s, **kwargs):
    ''' Independent variable function that returns Z coordinate \
    of spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    Z coordinate of state vector s.
    '''    
    return s[2]

def iVarVX(t, s, **kwargs):
    ''' Independent variable function that returns VX coordinate \
    of spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    VX coordinate of state vector s.
    '''    
    return s[3]

def iVarVY(t, s, **kwargs):
    ''' Independent variable function that returns VY coordinate \
    of spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    VY coordinate of state vector s.
    '''
    return s[4]

def iVarVZ(t, s, **kwargs):
    ''' Independent variable function that returns VZ coordinate \
    of spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    VZ coordinate of state vector s.
    '''
    return s[5]

def iVarT(t, s, **kwargs):
    ''' Independent variable function that returns time corresponds to\
    spacecraft state vector.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    Time corresponds to state vector s.
    '''    
    return t

def iVarR(t, s, **kwargs):
    ''' Independent variable function that returns length of radius-vector \
    calculated relative to specified center.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
    center : array-like with 3 components
        Coordinates of center relative to which radius-vector will be found
        
    Returns
    -------
        R : scalar
            Length of radius-vector
    '''    
    center = kwargs.get('center', np.zeros(3))
    return math.sqrt((s[0]-center[0])**2+(s[1]-center[1])**2+(s[2]-center[2])**2)

def iVarR2(t, s, **kwargs):
    ''' Independent variable function that returns length squared of \
    radius-vector calculated relative to specified center.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
    center : array-like with 3 components
        Coordinates of center relative to which radius-vector will be found
        
    Returns
    -------
        R2 : scalar
            Length squared of radius-vector
    '''    
    center = kwargs.get('center', np.zeros(3))
    return (s[0]-center[0])**2+(s[1]-center[1])**2+(s[2]-center[2])**2

def iVarAlpha(t, s, **kwargs):
    ''' Independent variable function that returns angle Alpha \
    in radians calculated relative to specified center point at X-axis. \
    Positive direction of angle is counter clockwise with zero at X-axis.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
    center : scalar
        Coordinate of center point relative to which angle will be
        calculated
        
    Returns
    -------
    
    alpha : scalar
        Angle in radians calculated relative to specified center 
        point at X-axis.
    '''    
    return math.atan2(s[1], s[0]-kwargs.get('center', 0))

def iVarRdotV(t, s, **kwargs):
    ''' Independent variable function that returns dot (scalar) product \
    of radius-vector relative to specified center by velocity vector of \
    spacecraft. This variable reaches zero in PERIGEE and APOGEE relative \
    to center.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
    center : array-like with 3 components
        Coordinates of center relative to which dot product will be found
        
    Returns
    -------
    
    RdotV : scalar
        Dot product of radius-vector relative to specified center
        by velocity vector of spacecraft.
    '''     
    center = kwargs.get('center', np.zeros(3))
    r = s[:3] - center
    v = s[3:6]
    return r[0]*v[0]+r[1]*v[1]+r[2]*v[2]

def stopFun(t, s, lst, ivar=iVarY, stopval=0, direction=0, corr = True, **kwargs):
    ''' Universal event detection function for scipy.integrate.ode \
        solout application. Provides termination of integration process \
        when some event occurs. This happens when independent variable \
        goes through defined stopval value in specified direction.
        Uses almost the same ideas as in matlab event functions.
        Can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation)
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst.
        
    ivar : function(t, s, **kwargs)
        Should return independent variable from spacecraft state vector.
        
    stopval : double
        stopFun return -1 if independent variable crosses stopval value in
        right direction
        
    direction : integer
         1 : stops integration when independent variable crosses stopval value
             from NEGATIVE to POSITIVE values
        -1 : stops integration when independent variable crosses stopval value
             from POSITIVE to NEGATIVE values
         0 : in both cases
         (like 'direction' argument in matlab's event functions)
         
    corr : bool
        Determines whether it is necessary to adjust last state vector or not
                      
    Returns
    -------
    
    -1 : scalar
        When y coordinate of spacecraft crosses zero. Will be treated by
        scipy.integrate.ode as it should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should NOT
        stop integration process.
        
          
    '''
    cur_iv = ivar(t, s, **kwargs)
    if not lst: # fast way to check if lst is empty
        lst.append(np.asarray([*s,t,cur_iv]))
        return 0
    lst.append(np.asarray([*s,t,cur_iv]))
    prev_s = lst[-2]
    prev_iv = prev_s[-1]
    f1 = (prev_iv < stopval) and (cur_iv > stopval) and ((direction == 1) or (direction == 0))
    f2 = (prev_iv > stopval) and (cur_iv < stopval) and ((direction == -1) or (direction == 0))
    if f1 or f2:
        if corr:
            arr = np.asarray(lst)
            interp = interp1d(arr[-4:,-1], arr[-4:], axis=0, kind='cubic', copy=False, assume_sorted=False)
            last_s = interp(stopval)
            lst.pop()
            lst.append(np.asarray([*last_s]))
        return -1
    return 0    



def stopFunCombined(t, s, lst, events, out=[], **kwargs):
    ''' Universal event detection function that handles multiple events \ 
        for scipy.integrate.ode solout application. Provides termination \
        of integration process when terminate event(s) occur. This happens \
        when independent variable goes through defined stopval value in \
        specified direction.
        Uses almost the same ideas as in matlab event functions.
        Can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation)
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst.
        
    events : list of dicts
        Each dict consists of necessary information for event:
        {
        
        ivar : function(t, s, **kwargs)
            Should return independent variable from spacecraft state vector.
        
        stopval : double
            stopFun return -1 if independent variable crosses stopval value in
            right direction
        
        direction : integer
            1 : stops integration when independent variable crosses stopval value
                 from NEGATIVE to POSITIVE values
            -1 : stops integration when independent variable crosses stopval value
                 from POSITIVE to NEGATIVE values
            0 : in both cases
                 (like 'direction' argument in matlab's event functions)
                 
        isterminal : integer, bool         
            Terminal event terminates integration process when event occurs.
            
        corr : bool
            Determines whether it is necessary to adjust last state vector or not
        
        kwargs : dict
            Other parameters for ivar function
        }
              
    Returns
    -------
    
    -1 : scalar
        When independent variable goes through defined stopval value in \
        specified direction. Will be treated by scipy.integrate.ode as it \
        should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should NOT
        stop integration process.
        
          
    '''
    if not events:
        return 0
    
    terminal = False
    cur_ivs = []
    sn = s.shape[0] + 1
    
    trm_evs = []
    
    for event in events:
        ivar = event['ivar']
        evkwargs = event.get('kwargs', {})        
        cur_iv = ivar(t, s, **evkwargs)
        cur_ivs.append(cur_iv)

    if not lst: # fast way to check if lst is empty
        lst.append(np.asarray([*s,t,*cur_ivs]))
        return 0
        
    lst.append(np.asarray([*s,t,*cur_ivs]))

    for i, event in enumerate(events):
        stopval = event.get('stopval', 0)
        direction = event.get('direction', 0)
        corr = event.get('corr', True)
        isterminal = event.get('isterminal', True)

        cur_iv = cur_ivs[i]
        prev_iv = lst[-2][sn+i]

        f1 = (prev_iv < stopval) and (cur_iv > stopval) and ((direction == 1) or (direction == 0))
        f2 = (prev_iv > stopval) and (cur_iv < stopval) and ((direction == -1) or (direction == 0))
        if f1 or f2:
            last_s = lst[-1]
            if corr:
                # calculation of corrected state vector using cubic spline interpolation
                # print('[%d]%f<>%f<>%f' % (i, prev_iv, stopval, cur_iv))
                arr = np.asarray(lst)
                interp = interp1d(arr[-4:,sn+i], arr[-4:], axis=0, kind='cubic', copy=False, assume_sorted=False)
                last_s = interp(stopval)
            out.append([i, last_s])
            if isterminal:
                terminal = True
                if corr:
                    trm_evs.append(last_s)

    if terminal:
        if corr:
            # correction of last (extended) state vector with state vector of
            # last terminal event occured
            if trm_evs:
                # math.abs is needed for backward integration (negative time)
                last_trm_ev = max(trm_evs, key=lambda x: math.fabs(x[sn-1]))
                lst.pop()
                lst.append(last_trm_ev)
        return -1
    
    return 0

def accurateEvent(arr, stopval=0):
    ''' DEPRECATED
        WAS INCORPORATED IN stopFun AND stopFunCombined
        Accurate calculation of spacecraft state vector at event \
        right after the completion of integration process by terminating it \
        with stopFun. Uses last component of extended state vector \
        (independent variable).
    
    Parameters
    ----------

    arr : numpy array of n-by-k shape, where n > 3
        Array of state vectors. When stopFun was used for CRTBP problem
        it have size n-by-8 (x, y, z, vx, vy, vz, t, iv)
      
    stopval : double
        Same value that was used in stopFun for event detection
                             
    Returns
    -------
    
    s : numpy array of k elements
        State vector at event calculted with third order spline interpolation        
          
    '''
    if arr.shape[0] < 4:
        return arr[-1]
    interp = interp1d(arr[-4:,-1], arr[-4:], axis=0, kind='cubic', copy=False, assume_sorted=False)
    return interp(stopval)


'''
    OLD FUNCTIONS
'''

def stopNull(t, s, lst, **kwargs):
    ''' Dummy function for scipy.integrate.ode solout application. \
        Can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst. 
                      
    Returns
    -------
    
    0 : scalar
        Always. Will be treated by scipy.integrate.ode as it shouldn't
        stop integration process.
          
    '''
    lst.append(np.hstack((s,t)))
    return 0

def stopY0(t, s, lst, **kwargs):
    ''' Solout function for scipy.integrate.ode. Stops integration \
        when Y coordinate of spacecraft goes through zero.
        Also can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst.
        
    direction : integer
         1 : stops integration when y crosses zero from negative to positive values
        -1 : stops integration when y crosses zero from positive to negative values
         0 : in both cases
         (like 'direction' argument in matlab's event functions)


    Returns
    -------
    
    -1 : scalar
        When y coordinate of spacecraft crosses zero. Will be treated by
        scipy.integrate.ode as it should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should NOT
        stop integration process.
          
    '''
    lst.append(np.hstack((s,t)))
    if len(lst) == 1:
        return 0
    prev_s1 = lst[-2][1]
    direction = kwargs.get('direction', 0)
    if ((prev_s1 < 0) and (s[1] > 0) and ((direction == 1) or (direction == 0))):
        return -1
    if ((prev_s1 > 0) and (s[1] < 0) and ((direction == -1) or (direction == 0))):
        return -1
    return 0

def stopAlpha(t, s, lst, **kwargs):
    ''' Solout function for scipy.integrate.ode. Stops integration \
        when angle measured from X-axis at counter clockwise direction \
        crosses alpha value.
        Also can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst.
        
    center : double
        X coordinate of libration point or another center point.
    
    alpha : double
        Termination angle relative to center. Angle will be measured
        from X-axis at counter clockwise direction.

    direction : integer
         1 : stops integration when angle crosses alpha value from negative to positive values
        -1 : stops integration when angle crosses alpha value from positive to negative values
         0 : in both cases
         (like 'direction' argument in matlab's event functions)

    Returns
    -------
    
    -1 : scalar
        If Y coordinate of spacecraft < 0. Will be treated by
        scipy.integrate.ode as it should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should NOT
        stop integration process.
          
    '''
    lst.append(np.hstack((s,t)))
    if len(lst) == 1:
        return 0
    center = kwargs.get('center', 0)
    angle = math.radians(kwargs.get('angle', 0))
    prev_angle = math.atan2(lst[-2][1], lst[-2][0]-center)
    cur_angle = math.atan2(s[1], s[0]-center)
    direction = kwargs.get('direction', 0)
    if ((prev_angle < angle) and (cur_angle > angle) and ((direction == 1) or (direction == 0))):
        return -1
    if ((prev_angle > angle) and (cur_angle < angle) and ((direction == -1) or (direction == 0))):
        return -1
    return 0


def stopY0m(t, s, lst, **kwargs):
    ''' Solout function for scipy.integrate.ode. Stops integration \
        when Y coordinate of spacecraft becomes lower than zero.
        Also can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst. 
                      
    Returns
    -------
    
    -1 : scalar
        If Y coordinate of spacecraft < 0. Will be treated by
        scipy.integrate.ode as it should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should NOT
        stop integration process.
          
    '''
    lst.append(np.hstack((s,t)))
    if (s[1] < 0):
        return -1
    return 0

def stopPlanes(t, s, lst, **kwargs):
    ''' Solout function for scipy.integrate.ode. Stops integration \
        when spacecraft reaches any of 2 planes.
        Also can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst. 
        
    planes : array_like of 2 scalars
        Defines planes x == planes[0] and x == planes[1] which crossing
        by spacecraft will stop integration.
        
                      
    Returns
    -------
    
    -1 : scalar
        When spacecraft reaches planes. Will be treated by
        scipy.integrate.ode as it should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should NOT
        stop integration process.
                 
    '''
    lst.append(np.hstack((s,t)))
    if ((s[0] < kwargs['planes'][0]) or (s[0] > kwargs['planes'][1])):
        return -1
    return 0

def stop3Planes(t, s, lst, **kwargs):
    ''' Solout function for scipy.integrate.ode. Stops integration \
        when spacecraft reaches any of 3 planes.
        Also can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst. 
        
    planes : array_like of 3 scalars
        Defines planes x == planes[0], x == planes[1] and
        |y| == planes[2] which crossing by spacecraft will stop integration.

    Returns
    -------
    
    -1 : scalar
        When spacecraft reaches any of planes. Will be treated by
        scipy.integrate.ode as it should STOP integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should
        CONTINUE integration process.
        
    See Also
    --------
    
    crtbp_prop.prop2Planes.
          
    '''
    lst.append(np.hstack((s,t)))
    if ((s[0] < kwargs['planes'][0]) or (s[0] > kwargs['planes'][1]) or (math.fabs(s[1]) > kwargs['planes'][2])):
        return -1
    return 0

def stopSpheres(t, s, lst, **kwargs):
    ''' Solout function for scipy.integrate.ode. Stops integration \
        when spacecraft reaches any of 2 spheres.
        Also can be used for gathering all intergation steps.
        Shoudn't be called directly but through scipy.integrate.ode.
    
    Parameters
    ----------

    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)

    lst : list
        Every call of this function put np.hstack of (s, t) into lst. 
        
    mu : scalar
        mu = mu1 = m1 / (m1 + m2), 
        where m1 and m2 - masses of two main bodies, m1 > m2
        Used as position of small body in this function.
    
    spheres : array_like of 2 scalars
        Defines spheres with centers in small body and radiuses r == spheres[0] and r == spheres[1]
        which crossing by spacecraft will stop integration.
        
                      
    Returns
    -------
    
    -1 : scalar
        When spacecraft reaches any of spheres. Will be treated by
        scipy.integrate.ode as it should STOP integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it should
        CONTINUE integration process.
                 
    '''
    lst.append(np.hstack((s,t)))
    ds = s[:3].copy() # take only coordinates 0,1,2
    ds[0] -= kwargs['mu'] # subtract small body position
    r = ds[0]**2+ds[1]**2+ds[2]**2 # calculate radius relative to small body
    r0 = kwargs['spheres'][0]**2
    r1 = kwargs['spheres'][1]**2
    if ((r < r0) or (r > r1)):
        return -1
    return 0