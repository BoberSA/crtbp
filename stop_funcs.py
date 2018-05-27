# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:55:16 2017

@author: Stanislav Bober
"""

import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from crtbp_ode import crtbp
#from numba import njit
import matplotlib.pyplot as plt

'''
    NEW FUNCTIONS 16-07-2017
'''

#def sign(x):
#    return (1.0 if (x) > 0.0 else (-1.0 if (x) < 0.0 else 0.0))


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

def iVarAX(t, s, **kwargs):
    ''' Independent variable function that returns AX - projection of \
    acceleration vector to X axis.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    AX - projection of acceleration vector to X axis
    '''
#    print('a', a)
    a = crtbp(t, s, kwargs['mu'])
    return a[3]

def iVarAY(t, s, **kwargs):
    ''' Independent variable function that returns AY - projection of \
    acceleration vector to Y axis.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    AY - projection of acceleration vector to Y axis
    '''
    return crtbp(t, s, kwargs['mu'])[4]

def iVarAZ(t, s, **kwargs):
    ''' Independent variable function that returns AZ - projection of \
    acceleration vector to Z axis.
    Should be used in stopFun and accurateEvent functions
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation)        
    s : array_like with 6 components
        State vector of massless spacecraft (x,y,z,vx,vy,vz)
        
    Returns
    -------
    AZ - projection of acceleration vector to Z axis
    '''
    return crtbp(t, s, kwargs['mu'])[5]

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

def iVarDR(t, s, **kwargs):
    ''' Independent variable function that returns acceleration value \
    along radius-vector calculated relative to specified center.
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
        A : scalar
            Acceleration value along radius-vector
    '''
    # projection of acceleration vector to radius-vector
    v = s[3:]   
    r = s[:3] - kwargs.get('center', np.zeros(3))
    r = r / math.sqrt(r[0]**2+r[1]**2+r[2]**2)
    return r[0]*v[0]+r[1]*v[1]+r[2]*v[2]

def iVarR2(t, s, **kwargs):
    ''' Independent variable function that returns length squared of \
    radius-vector calculated relative to specified center.
    Should be used in stopFunCombined.
    
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
    ''' Independent variable function that returns angle Alpha between \
    direction to SC in XY plane and X axis. Alpha measured in radians and \
    calculated relative to specified center point at X-axis. \
    Positive direction of angle is counter clockwise with zero at X-axis.
    Should be used in stopFunCombined.
    
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

def iVarDAlpha(t, s, **kwargs):
    ''' Independent variable function that returns angle Alpha between \
    direction to SC in XY plane and X axis. Alpha measured in radians and \
    calculated relative to specified center point at X-axis. \
    Positive direction of angle is counter clockwise with zero at X-axis.
    Should be used in stopFunCombined.
    
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
    # omega = cross(r, v)/dot(r,r)
    v = s[3:5]
    r = [s[0]-kwargs.get('center', 0), s[1]]
    omega = (r[0]*v[1]-r[1]*v[0])/(r[0]**2+r[1]**2)
    return omega

def iVarAlpha2(t, s, **kwargs):
    ''' Independent variable function that returns angle Alpha between \
    direction to SC in XY plane and -X axis. Alpha measured in radians and \
    calculated relative to specified center point at X-axis. \
    Positive direction of angle is counter clockwise with zero at X-axis.
    Should be used in stopFunCombined.
    
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
    return math.atan2(s[1], kwargs.get('center', 0)-s[0])

def iVarConeX(t, s, **kwargs):
    ''' Independent variable function that returns angle Alpha that defines \
    a cone with main axis along X direction. Alpha measured in radians  and \
    calculated relative to specified center point at X-axis. \
    Positive direction of angle is counter clockwise with zero at X-axis.
    Should be used in stopFunCombined.
    
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
        Angle that defines cone in X direction in radians calculated \
        relative to specified center point at X-axis.
    '''    
    return math.atan2(math.sqrt(s[1]**2+s[2]**2), s[0]-kwargs.get('center', 0))

def iVarRdotV(t, s, **kwargs):
    ''' Independent variable function that returns dot (scalar) product \
    of radius-vector relative to specified center by velocity vector of \
    spacecraft. This variable reaches zero in PERICENTER and APOCENTER \
    relative to center point.
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
#    r = r / (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
    v = s[3:6]
#    v = v / (v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
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


def stopFunCombinedInterp(t, s, lst, events, out=[], **kwargs):
    ''' Universal event detection function that handles multiple events. 
        Intended for scipy.integrate.ode solout application. Provides \
        termination of integration process when first terminate event occur. \
        This happens when independent variable associated with this event \
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
        Every call of this function put [*s,t,*cur_ivs] into lst, where 
        t - time (at current integration step),
        s - spacecraft state vector at time t,
        cur_ivs - list of independent variable values at time t.
        
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
        
        count : int
            Number of event occasions.
            If count == 0
            Then event doesnt occur (turned off event).
            
            If count == -1
            Then (possibly) unlimited number of events can occur.
            
            If isterminal == True
                If count == 1
                Then only one terminal event (possibly) occur.
                If count > 1
                Then non-terminal event (possibly) triggers count-1
                times and one terminal event (possibly) occur.
            If isterminal == False
                Event (possibly) occurs count times.

        kwargs : dict
            Other parameters for ivar function
        }
        
    out : list
        If non-terminal event(s) occur in [ti-1, ti] interval, 'out' will
        be filled with [ei, ci, np.array([*s,te,*cur_ivs])], where:
        ei - event index in events list,
        ci - event triggered ci times,
        te - time of event,
        s - state vector at te,
        cur_ivs - values of independent variables at te.
              
    Returns
    -------
    
    -1 : scalar
        When there are terminal event in event list and independent variable \
        assiciated with this event goes through defined stopval value in \
        specified direction. Will be treated by scipy.integrate.ode as it \
        should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it\
        should continue integration process.
        
          
    '''
    if not events:
        return 0
    
    terminal = False
    cur_ivs = []
    sn = s.shape[0] + 1
    
    trm_evs = []
    out_ = []
    
    for event in events:
        ivar = event['ivar']
        evkwargs = event.get('kwargs', {})        
        cur_iv = ivar(t, s, **evkwargs)
        cur_ivs.append(cur_iv)

    if not lst: # fast way to check if lst is empty
        cur_cnt = []
        for event in events:
            cur_cnt.append(event.get('count', -1))
        if not out:
            out.append(cur_cnt)
        else:
            out[0] = cur_cnt
        lst.append(np.asarray([*s,t,*cur_ivs]))
        return 0
        
    lst.append(np.asarray([*s,t,*cur_ivs]))

    cur_cnt = out[0]
    for i, event in enumerate(events):
        stopval = event.get('stopval', 0)
        direction = event.get('direction', 0)
        corr = event.get('corr', True)
        isterminal = event.get('isterminal', True)
        init_cnt = event.get('count', -1)
        
        cur_iv = cur_ivs[i]
        prev_iv = lst[-2][sn+i]

        f1 = (prev_iv < stopval) and (cur_iv > stopval) and ((direction == 1) or (direction == 0))
        f2 = (prev_iv > stopval) and (cur_iv < stopval) and ((direction == -1) or (direction == 0))
        if (f1 or f2) and ((cur_cnt[i] == -1) or (cur_cnt[i] > 0)):
            if cur_cnt[i] > 0:
                cur_cnt[i] -= 1
            last_s = lst[-1].copy() # FIX: .copy() is important here
            if corr:
                # calculation of corrected state vector using cubic spline interpolation
                # if number of state vectors available are less than 4 then
                # quadratic/linear interpolation are used
                # print('[%d]%f<>%f<>%f' % (i, prev_iv, stopval, cur_iv))
                arr = np.asarray(lst)
                if arr.shape[0] == 2:
                    interp = interp1d(arr[-2:,sn+i], arr[-2:], axis=0, kind='linear', copy=False, assume_sorted=False)
                elif arr.shape[0] == 3:
                    interp = interp1d(arr[-3:,sn+i], arr[-3:], axis=0, kind='quadratic', copy=False, assume_sorted=False)
                else:
                    interp = interp1d(arr[-4:,sn+i], arr[-4:], axis=0, kind='cubic', copy=False, assume_sorted=False)                    
                last_s = interp(stopval)
            out_.append([i, (-1 if cur_cnt[i]==-1 else init_cnt-cur_cnt[i]), last_s])
            if isterminal and ((cur_cnt[i] == -1) or (cur_cnt[i] == 0)):
                terminal = True
                if corr:
                    trm_evs.append(last_s)
    
    if out_:
        tsort = kwargs.get('tsort', True)
        # extend 'out' list with
        if tsort: #sorted by time events
            out.extend(sorted(out_, key=lambda x: math.fabs(x[2][sn-1])))
        else: #sorted by index in event list
            out.extend(out_)
        
    if terminal:
        if corr:
            # correction of last state vector with state vector of
            # first terminal event occured
            if trm_evs:
                # math.abs is needed for backward integration (negative time)
                last_trm_ev = min(trm_evs, key=lambda x: math.fabs(x[sn-1]))
                lst.pop()
                lst.append(last_trm_ev)
#            out.pop(0)
        return -1
    
    return 0

def stopFunCombined(t, s, lst, events, out=[], **kwargs):
    ''' Universal event detection function that handles multiple events. 
        Intended for scipy.integrate.ode solout application. Provides \
        termination of integration process when first terminate event occur. \
        This happens when independent variable associated with this event \
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
        Every call of this function put [*s,t,*cur_ivs] into lst, where 
        t - time (at current integration step),
        s - spacecraft state vector at time t,
        cur_ivs - list of independent variable values at time t.
        
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
        
        count : int
            Number of event occasions.
            If count == 0
            Then event doesnt occur (turned off event).
            
            If count == -1
            Then (possibly) unlimited number of events can occur.
            
            If isterminal == True
                If count == 1
                Then only one terminal event (possibly) occur.
                If count > 1
                Then non-terminal event (possibly) triggers count-1
                times and one terminal event (possibly) occur.
            If isterminal == False
                Event (possibly) occurs count times.

        kwargs : dict
            Other parameters for ivar function
        }
        
    out : list
        If non-terminal event(s) occur in [ti-1, ti] interval, 'out' will
        be filled with [ei, ci, np.array([*s,te,*cur_ivs])], where:
        ei - event index in events list,
        ci - event triggered ci times,
        te - time of event,
        s - state vector at te,
        cur_ivs - values of independent variables at te.
              
    Returns
    -------
    
    -1 : scalar
        When there are terminal event in event list and independent variable \
        assiciated with this event goes through defined stopval value in \
        specified direction. Will be treated by scipy.integrate.ode as it \
        should stop integration process.
        
    0 : scalar
        Otherwise. Will be treated by scipy.integrate.ode as it\
        should continue integration process.
        
          
    '''
    if not events:
        return 0
    
    terminal = False
    cur_ivs = []
    sn = s.shape[0] + 1
    
    trm_evs = []
    out_ = []
    
    for event in events:
        ivar = event['ivar']
        evkwargs = event.get('kwargs', {})        
        cur_iv = ivar(t, s, **evkwargs)
        cur_ivs.append(cur_iv)

    if not lst: # fast way to check if lst is empty
        cur_cnt = []
        for event in events:
            cur_cnt.append(event.get('count', -1))
        if not out:
            out.append(cur_cnt)
        else:
            out[0] = cur_cnt
#        lst.append(np.asarray([*s,t,*cur_ivs]))
        lst.append([*s,t,*cur_ivs])
        return 0
        
#    lst.append(np.asarray([*s,t,*cur_ivs]))
    lst.append([*s,t,*cur_ivs])

    cur_cnt = out[0]
    for i, event in enumerate(events):
        stopval = event.get('stopval', 0)
        direction = event.get('direction', 0)
        corr = event.get('corr', True)
        isterminal = event.get('isterminal', True)
        init_cnt = event.get('count', -1)
        
        cur_iv = cur_ivs[i]
        prev_iv = lst[-2][sn+i]

        f1 = (prev_iv < stopval) and (cur_iv > stopval) and ((direction == 1) or (direction == 0))
        f2 = (prev_iv > stopval) and (cur_iv < stopval) and ((direction == -1) or (direction == 0))
        if (f1 or f2) and ((cur_cnt[i] == -1) or (cur_cnt[i] > 0)):
            if cur_cnt[i] > 0:
                cur_cnt[i] -= 1
            last_s = lst[-1].copy() # FIX: .copy() is important here
            corr_f = True # if correction process converged between lst[-2] and lst[-1]
            if corr:
                # calculation of corrected state vector using Newton's method
                # print('[%d]%f<>%f<>%f' % (i, prev_iv, stopval, cur_iv))
                
                prop = ode(crtbp)
                prop.set_f_params(*[kwargs['mu']])
                method = kwargs['int_param'].get('method', 'dopri5')
                prop.set_integrator(method, **kwargs['int_param'])
                atol = kwargs['int_param']['atol']
#                print('SN', sn)
                s_n = lst[-2][:sn-1]
                tn = lst[-2][sn-1]
                maxdt = lst[-1][sn-1]-tn
                ivar = event['ivar']
                dvar = event['dvar']
                evkwargs = event.get('kwargs', {})        
                cur_iv_ = prev_iv
                
                j = 0
                maxit = kwargs.get('maxit', 100)
                while math.fabs(cur_iv_ - stopval) > atol and j < maxit:
                    prop.set_initial_value(s_n, tn)
                    cur_dt = (cur_iv_-stopval)/dvar(tn, s_n, **evkwargs)
                    # in case of very big steps (when dvar are very small value)
                    if ((tn - cur_dt) < lst[-2][sn-1]) or ((tn - cur_dt) > lst[-1][sn-1]):
                        cur_dt = -maxdt*0.5
                    tn -= cur_dt
                    s_n = prop.integrate(tn)
                    cur_iv_ = ivar(tn, s_n, **evkwargs)
                    j += 1
                    
                if j == maxit:
                    print('Newton alarm')
                
                if ((tn >= lst[-2][sn-1]) and (tn <= lst[-1][sn-1])) or \
                   ((tn <= lst[-2][sn-1]) and (tn >= lst[-1][sn-1])):                
                    cur_ivs_ = []
                    for event_ in events:
                        cur_ivs_.append(event_['ivar'](tn, s_n, **event_.get('kwargs', {})))               
                
#                    last_s = np.array([*s_n,tn,*cur_ivs_])
                    last_s = [*s_n,tn,*cur_ivs_]
                else:
                    corr_f = False # correction process doesn't converged between lst[-2] and lst[-1]
                    print('Newton correction is out of range')
                    #TODO something with cur_cnt
            if corr_f:
                out_.append([i, (-1 if cur_cnt[i]==-1 else init_cnt-cur_cnt[i]), last_s])
                if isterminal and ((cur_cnt[i] == -1) or (cur_cnt[i] == 0)):
                    terminal = True
                    if corr:
                        trm_evs.append(last_s)
    
    if out_:
        tsort = kwargs.get('tsort', True)
        # extend 'out' list with
        if tsort: #sorted by time events
            out.extend(sorted(out_, key=lambda x: math.fabs(x[2][sn-1])))
        else: #sorted by index in event list
            out.extend(out_)
        
    if terminal:
        if corr:
            # correction of last state vector with state vector of
            # first terminal event occured
            if trm_evs:
                # math.abs is needed for backward integration (negative time)
                last_trm_ev = min(trm_evs, key=lambda x: math.fabs(x[sn-1]))
                lst.pop()
                lst.append(last_trm_ev)
#            out.pop(0)
        return -1
    
    return 0

def stopFunCombined_debug(t, s, lst, events, out=[], **kwargs):
    ''' Same as stopFunCombined but for debug purposes '''

    if not events:
        return 0
    
    terminal = False
    cur_ivs = []
    sn = s.shape[0] + 1
    
    trm_evs = []
    out_ = []
    
    for event in events:
        ivar = event['ivar']
        evkwargs = event.get('kwargs', {})        
        cur_iv = ivar(t, s, **evkwargs)
        cur_ivs.append(cur_iv)

    if not lst: # fast way to check if lst is empty
        cur_cnt = []
        for event in events:
            cur_cnt.append(event.get('count', -1))
        if not out:
            out.append(cur_cnt)
        else:
            out[0] = cur_cnt
#        lst.append(np.asarray([*s,t,*cur_ivs]))
        lst.append([*s,t,*cur_ivs])
        return 0
        
#    lst.append(np.asarray([*s,t,*cur_ivs]))
    lst.append([*s,t,*cur_ivs])

    cur_cnt = out[0]
    for i, event in enumerate(events):
        stopval = event.get('stopval', 0)
        direction = event.get('direction', 0)
        corr = event.get('corr', True)
        isterminal = event.get('isterminal', True)
        init_cnt = event.get('count', -1)
        
        cur_iv = cur_ivs[i]
        prev_iv = lst[-2][sn+i]

        f1 = (prev_iv < stopval) and (cur_iv > stopval) and ((direction == 1) or (direction == 0))
        f2 = (prev_iv > stopval) and (cur_iv < stopval) and ((direction == -1) or (direction == 0))
        if (f1 or f2) and ((cur_cnt[i] == -1) or (cur_cnt[i] > 0)):
            if cur_cnt[i] > 0:
                cur_cnt[i] -= 1
            last_s = lst[-1].copy() # FIX: .copy() is important here
            corr_f = True # if correction process converged between lst[-2] and lst[-1]
            if corr:
                # calculation of corrected state vector using Newton's method
                # print('[%d]%f<>%f<>%f' % (i, prev_iv, stopval, cur_iv))
                
                prop = ode(crtbp)
                prop.set_f_params(*[kwargs['mu']])
                method = kwargs['int_param'].get('method', 'dopri5')
                prop.set_integrator(method, **kwargs['int_param'])
                atol = kwargs['int_param']['atol']
#                print('SN', sn)
                s_n = lst[-2][:sn-1]
                tn = lst[-2][sn-1]
                maxdt = lst[-1][sn-1]-tn
                ivar = event['ivar']
                dvar = event['dvar']
                evkwargs = event.get('kwargs', {})        
                cur_iv_ = prev_iv
                
                j = 0
                maxit = kwargs.get('maxit', 100)
                #debug
                debug_lst = [[cur_iv_, tn]]
                #
                while math.fabs(cur_iv_ - stopval) > atol and j < maxit:
                    prop.set_initial_value(s_n, tn)
                    cur_dt = (cur_iv_-stopval)/dvar(tn, s_n, **evkwargs)
                    # in case of very big steps (when dvar are very small value)
                    if ((tn - cur_dt) < lst[-2][sn-1]) or ((tn - cur_dt) > lst[-1][sn-1]):
                        cur_dt = -maxdt*0.5
                    tn -= cur_dt
                    s_n = prop.integrate(tn)
                    cur_iv_ = ivar(tn, s_n, **evkwargs)
                    #debug
                    debug_lst.append([cur_iv_, tn])
                    #
                    j += 1
                
                if j == maxit:
                    print('Newton alarm')
                
                if ((tn >= lst[-2][sn-1]) and (tn <= lst[-1][sn-1])) or \
                   ((tn <= lst[-2][sn-1]) and (tn >= lst[-1][sn-1])):                
                    cur_ivs_ = []
                    for event_ in events:
                        cur_ivs_.append(event_['ivar'](tn, s_n, **event_.get('kwargs', {})))               
                
#                    last_s = np.array([*s_n,tn,*cur_ivs_])
                    last_s = [*s_n,tn,*cur_ivs_]
                else:
                    corr_f = False # correction process doesn't converged between lst[-2] and lst[-1]
                    print('Newton correction is out of range')
                    #TODO something with cur_cnt
                    #debug
                    print('i', i)
                    print('event', event)
                    debug_arr = np.array(debug_lst)
                    plt.figure(figsize=(10,10))
                    plt.plot([lst[-2][sn-1], lst[-1][sn-1]], [stopval]*2, 'r')
                    plt.plot(debug_arr[:,1], debug_arr[:,0], '.-b')
                    for jj, dbg_pt in enumerate(debug_arr):
                        plt.text(dbg_pt[1], dbg_pt[0], '%d'%jj)
                    plt.show()
                    #    
            if corr_f:
                out_.append([i, (-1 if cur_cnt[i]==-1 else init_cnt-cur_cnt[i]), last_s])
                if isterminal and ((cur_cnt[i] == -1) or (cur_cnt[i] == 0)):
                    terminal = True
                    if corr:
                        trm_evs.append(last_s)
    
    if out_:
        tsort = kwargs.get('tsort', True)
        # extend 'out' list with
        if tsort: #sorted by time events
            out.extend(sorted(out_, key=lambda x: math.fabs(x[2][sn-1])))
        else: #sorted by index in event list
            out.extend(out_)
        
    if terminal:
        if corr:
            # correction of last state vector with state vector of
            # first terminal event occured
            if trm_evs:
                # math.abs is needed for backward integration (negative time)
                last_trm_ev = min(trm_evs, key=lambda x: math.fabs(x[sn-1]))
                lst.pop()
                lst.append(last_trm_ev)
#            out.pop(0)
        return -1
    
    return 0



def calcEvents(arr, ti=6, stopf=stopFunCombined, **kwargs):
    ''' Calculate events for already calculated trajectory.
    
    Parameters
    ----------

    arr : numpy array of n-by-k shape
        Array of state vectors or extended state vectors.
      
    stopf : function
        Stop function for event calculation. Default: stopFunCombined.
                             
    kwargs : dict
        Additional keyworded arguments for stop function.

    Returns
    -------
        Only through 'out' argument of stop function. 
        
    Example
    -------
    
    evY0 = {'ivar':iVarY, 'stopval':  0, 'direction': 0, 'isterminal':False,  'corr':True}
    calcEvents(orb, out=evout, events=[evY0])
    
    Calculate all crossings of Y=0 plane for trajectory orb.
          
    '''    
    
    lst = []
    for s in arr:
        ret = stopf(s[ti], s[:ti], lst, **kwargs)
        if ret == -1:
            return


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