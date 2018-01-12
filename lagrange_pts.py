# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:54:06 2017

@author: Stanislav Bober
"""
import scipy.optimize
from crtbp_ode import crtbp

def lagrange1(mu2):
    mu1 = 1-mu2
    a = (mu2/(3*mu1))**(1/3)
    l1 = a-1/3*a**2-1/9*a**3-23/81*a**4
    return -l1 + mu1

def lagrange2(mu2):
    mu1 = 1-mu2
    a = (mu2/(3*mu1))**(1/3)
    l2 = a+1/3*a**2-1/9*a**3-31/81*a**4
    return l2 + mu1