import numpy as np

def Omega(s, mu):
    # mu = mu2
    # check!!!!
    x, y, x1, y1 = s
    
    
    return 0

def crtbp_planar(t, s, mu1):
    ''' Right part of Circular Restricted Three Body Problem ODE
        Dimensionless formulation.
        See Murray, Dermott 'Solar System Dynamics'.
    
    Parameters
    ----------
    t : scalar
        Dimensionless time (same as angle of system rotation).
        
    s : array_like with 4 components
        State vector of massless spacecraft (x,y,vx,vy)
        
    mu : scalar
         mu = mu1 = m1 / (m1 + m2), 
         where m1 and m2 - masses of two main bodies, m1 > m2
         
    Returns
    -------
    
    ds : np.array
        First order derivative with respect to time of spacecraft
        state vector (vx,vy,dvx,dvy)
    '''
    
    x, y, x1, y1 = s
    #mu1 = 1 - mu
    mu2 = 1-mu1
    mu = mu1
    
    y2 = y * y 
    #r13 = ((x-mu2)**2+y**2)**1.5
    #r23 = ((x+mu1)**2+y**2)**1.5
    r13 = ((x + mu2) * (x + mu2) + y2) ** 1.5;
    r23 = ((x - mu ) * (x - mu ) + y2) ** 1.5;

    yzcmn = (mu / r13 + mu2 / r23);
    #dx1dt = 2*y1 + x - (mu1*(x-mu2)/r13 + mu2*(x+mu1)/r23)
    #dy1dt = -2*x1 +y - y*(mu1/r13+mu2/r23)
    dx1dt =  2 * y1 + x - (mu * (x + mu2) / r13 + mu2 * (x - mu) / r23);
    dy1dt = -2 * x1 + y - yzcmn * y;
    
    return np.array([x1, y1, dx1dt, dy1dt])