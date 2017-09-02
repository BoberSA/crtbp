# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:54:06 2017

"""
import scipy.optimize
from crtbp_ode import crtbp

def lagrange1(mu):
    ''' Numerically calculate position of Lagrange L1 point.
        Uses scipy.optimize.root to find where acceleration in X direction
        becomes zero. Initial state vector [0.0, 0, 0, 0, 0, 0].
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.
                             
    Returns
    -------
    
    pos : scalar
        Dimensionless X coordinate of L1 point.
    
    See Also
    --------
    
    crtbp_ode.crtbp.

    '''
    #mu2 = 1 - mu
    #a = (mu2/(3*mu))**(1/3)
    #l1 = a-1/3*a**2-1/9*a**3-23/81*a**4
    #return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], mu-l1).x
    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], 0.).x[0]

def lagrange2(mu):
    ''' Numerically calculate position of Lagrange L2 point.
        Uses scipy.optimize.root to find where acceleration in X direction
        becomes zero. Initial state vector [2.0, 0, 0, 0, 0, 0].
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.
                             
    Returns
    -------
    
    pos : scalar
        Dimensionless X coordinate of L2 point.
        
    See Also
    --------
    
    crtbp_ode.crtbp.
          
    '''
    #mu2 = 1 - mu
    #a = (mu2/(3*mu))**(1/3)
    #l2 = a+1/3*a**2-1/9*a**3-31/81*a**4
    #return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], mu+l2).x
    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], 2.).x[0]
