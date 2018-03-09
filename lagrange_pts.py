# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:54:06 2017

@author: Stanislav Bober
"""
import scipy.optimize
import numpy as np
import math
from crtbp_ode import crtbp

def lagrange_pts(mu):
    ''' Calculate positions of all 5 Lagrange points.
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.
                             
    Returns
    -------
    
    L : numpy array of (5, 3) shape
        Dimensionless X, Y, Z coordinates of L1-L5 points.
    '''

    L = np.zeros((5, 3))
    L[0, 0] = lagrange1(mu)
    L[1, 0] = lagrange2(mu)
    L[2, 0] = lagrange3(mu)
    L[3, :2] = lagrange4(mu)
    L[4, :2] = lagrange5(mu)
    return L
    
    
def lagrange1(mu):
    ''' Numerically calculate position of Lagrange L1 point.
        Uses scipy.optimize.root to find where acceleration in X direction
        becomes zero. Initial state vector [0.5, 0, 0, 0, 0, 0].
    
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
#    mu2 = 1 - mu
#    a = (mu2/(3*mu))**(1/3)
#    l1 = a-1/3*a**2-1/9*a**3-23/81*a**4
#    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], mu-l1).x
    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], 0.5).x[0]

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
#    mu2 = 1 - mu
#    a = (mu2/(3*mu))**(1/3)
#    l2 = a+1/3*a**2-1/9*a**3-31/81*a**4
#    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], mu+l2).x
    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], 2.).x[0]

def lagrange3(mu):
    ''' Numerically calculate position of Lagrange L3 point.
        Uses scipy.optimize.root to find where acceleration in X direction
        becomes zero. Initial state vector [-1.0, 0, 0, 0, 0, 0].
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.
                             
    Returns
    -------
    
    pos : scalar
        Dimensionless X coordinate of L3 point.
        
    See Also
    --------
    
    crtbp_ode.crtbp.
          
    '''
    return scipy.optimize.root(lambda x:crtbp(0, [x, 0, 0, 0, 0, 0], mu)[3], -1.).x[0]

def lagrange4(mu):
    ''' Return position of Lagrange L4 point.
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.
                             
    Returns
    -------
    
    pos : numpy array of (2,) shape
        Dimensionless X and Y coordinats of L4 point.
          
    '''
    return np.array([mu-0.5, math.sqrt(3)*0.5])

def lagrange5(mu):
    ''' Return position of Lagrange L5 point.
    
    Parameters
    ----------

    mu : scalar
        CRTBP mu1 coefficient.
                             
    Returns
    -------
    
    pos : numpy array of (2,) shape
        Dimensionless X and Y coordinats of L5 point.
          
    '''
    return np.array([mu-0.5, -math.sqrt(3)*0.5])