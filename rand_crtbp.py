# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 01:22:37 2017

"""

#import numpy as np
from numpy.random import rand
import math

def randBallOne():
    d = rand(3) * 2 - 1
    while d[0]**2+d[1]**2+d[2]**2 > 1:
        d = rand(3) * 2 - 1
    return d

def randBallV(v, r):
    d = rand(3) * 2 - 1
    while d[0]**2+d[1]**2+d[2]**2 > 1:
        d = rand(3) * 2 - 1
    return v + d * r

def randBallPercent(v, r):
    d = rand(3) * 2 - 1
    while d[0]**2+d[1]**2+d[2]**2 > 1:
        d = rand(3) * 2 - 1
    nrm = math.sqrt(v[0]**2+v[1]**2+v[2]**2)
    return v + d * (nrm * r)

def randSphereOne():
    d = rand(3) * 2 - 1
    while d[0]**2+d[1]**2+d[2]**2 > 1:
        d = rand(3) * 2 - 1
    n = math.sqrt(d[0]**2+d[1]**2+d[2]**2)
    return d / n

def randSphere(v, r):
    d = rand(3) * 2 - 1
    while d[0]**2+d[1]**2+d[2]**2 > 1:
        d = rand(3) * 2 - 1
    n = math.sqrt(d[0]**2+d[1]**2+d[2]**2)
    return v + (r / n) * d
