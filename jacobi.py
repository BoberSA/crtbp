# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:22:19 2018

@author: Stanislav
"""
import numpy as np

def omega(mu1, y):
    ''' Calculate Jacobi constant for all spacecraft positions.
        Can be used for calculation zero-velocity surfaces.
    
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    y : numpy array of (N,M) size, where M >= 3
        Array of spacecraft state vectors.
                       
    Returns
    -------
    
    J : numpy array of (N,1) shape
        Jacobi integral value for each position with zero-velocity assumption.   
       
    '''
    mu2 = 1.0 - mu1
    r1 = np.sqrt((y[:,0] + mu2)**2 + y[:,1]**2 + y[:,2]**2)
    r2 = np.sqrt((y[:,0] - mu1)**2 + y[:,1]**2 + y[:,2]**2)
    return 0.5*(y[:,0]**2 + y[:,1]**2) + mu1 / r1 + mu2 / r2
    
def jacobi_const(mu1, y):
    ''' Calculate Jacobi constant for all spacecraft state vectors.
        Uses: omega
    Parameters
    ----------
    mu : scalar
        CRTBP mu1 coefficient.

    y : numpy array of (N,M) size, where M >= 6
        Array of spacecraft state vectors.
                       
    Returns
    -------
    
    J : numpy array of (N,1) shape
        Jacobi integral value for each state vector.   
       
    '''
    return 2*omega(mu1, y) - y[:,3]**2 - y[:,4]**2 - y[:,5]**2