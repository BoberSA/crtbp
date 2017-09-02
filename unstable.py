# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 01:11:07 2017

"""

import math
import numpy as np
from scipy.integrate import ode
from scipy.optimize import fminbound
from scipy.interpolate import interp1d
from crtbp_ode import crtbp
from crtbp_prop import propCrtbp
from stop_funcs import stopPlanes
import matplotlib.pyplot as plt

# быстрая функция расчета устойчивого направления
# dt - шаг интегрирования
# dv - величина возмущения скорости

def calcStableDirFast(mu, y0, dt=np.pi, dv=1e-5, **kwargs):
    ''' Fast stable direction calculation for given spacecraft \
        state vector in CRTBP.
    '''
    
    prop = ode(crtbp)
    prop.set_f_params(*[mu])
    if 'int_param' in kwargs:
        prop.set_integrator('dopri5', **kwargs['int_param'])
    else:
        prop.set_integrator('dopri5')
    prop.set_initial_value(y0, 0.0)
    c = prop.integrate(dt)

    def _goal(beta): #, y0=y0, c=c, dt=dt, dv=dv
        y1 = y0.copy()
        y1[3] += dv * math.cos(beta)
        y1[4] += dv * math.sin(beta)
        prop.set_initial_value(y1, 0.0)
        p = prop.integrate(dt)       
        return (p[3]-c[3])**2+(p[4]-c[4])**2 #(p[0]-c[0])**2+(p[1]-c[1])**2+
    
    return fminbound(_goal, 0, 2*np.pi, full_output=kwargs.get('full_output', False), xtol=kwargs.get('xtol', 1e-5))


# точная функция расчета устойчивого направления
# dt - шаг интегрирования
# dv - величина возмущения скорости

def calcUnstableDirPlanes(mu, y0, planes, dv=1e-5, **kwargs):
    ''' Precise stable direction calculation for given spacecraft \
        state vector in CRTBP using bounding planes.
    '''
    def _goal(beta): #, y0=y0, c=c, dt=dt, dv=dv
        y1 = y0.copy()
        y1[3] += dv * math.cos(beta)
        y1[4] += dv * math.sin(beta)
        arr = propCrtbp(mu, y1, [0, np.pi*100], stopf=stopPlanes, planes=planes, **kwargs)
        interp = interp1d(arr[-4:, 0], arr[-4:], axis=0, kind='cubic', copy=False, assume_sorted=False)
        j = 1
        if arr[-1,0] < planes[0]:
            j = 0
        arr[-1]=interp(planes[j])
        if 'debug' in kwargs:
            plt.plot(arr[:,0], arr[:,1])
        return arr[-1,6]
        
    return fminbound(_goal, 0, 2*np.pi, full_output=kwargs.get('full_output', False), xtol=kwargs.get('xtol', 1e-5))
