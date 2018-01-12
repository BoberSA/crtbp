import numpy as np
import scipy, math
from crtbp_planar_ode import crtbp_planar as crtbp
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
    vstart = y1[2:4].copy()
    dv = dv0
    dvtol = kwargs.get('dvtol', 1e-16)
    
    rads = math.radians(beta)
    beta_n = np.array([math.cos(rads), math.sin(rads)])
       
    p, _ = prop2Limits(mu, y1, lims, **kwargs)
    y1[2:4] = vstart + dv * beta_n
    p1, _ = prop2Limits(mu, y1, lims, **kwargs)
    
    if p == p1:
        dv = -dv
       
    v = dv        
    i = 0
    while math.fabs(dv) > dvtol and i < 100:
        y1[2:4] = vstart + v * beta_n
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
    vstart = y1[2:4].copy()
    dv = dv0
    dvtol = kwargs.get('dvtol', 1e-16)
    
    rads = math.radians(beta)
    beta_n = np.array([math.cos(rads), math.sin(rads)])
       
    p, _ = prop2Limits(mu, y1, lims, **kwargs)
    y1[2:4] = vstart + dv * beta_n
    p1, arr = prop2Limits(mu, y1, lims, **kwargs)
    
    if p == p1:
        dv = -dv
       
    v = dv  
    lst.append(v)
    
    if p1 == 0:
        plt.plot(arr[:,0],arr[:,1], 'b')
    else:
        plt.plot(arr[:,0],arr[:,1], 'r')

    while math.fabs(dv) > dvtol:
        y1[2:4] = vstart + v * beta_n
        p1, arr = prop2Limits(mu, y1, lims, **kwargs)
        if p1 == 0:
            plt.plot(arr[:,0],arr[:,1], 'b')
        else:
            plt.plot(arr[:,0],arr[:,1], 'r')
        
        if p1 != p:
            v -= dv
            dv *= 0.5

        v += dv
        lst.append(v)
    
    plt.savefig('debug '+datetime.now().isoformat().replace(':','-')+'.png')
        
#    print('%g'%v, end=' ')
    return v * beta_n
