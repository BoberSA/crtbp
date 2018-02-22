# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:24:39 2018

@author: Stanislav Bober

Based on code from here:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection as Line3DCollection

def make_segments(x, y, z=None):
    ''' Create numpy array of line segments from x, y (and z) coordinates,
        in the correct format for LineCollection (Line3DCollection).
       
    Parameters
    ----------
    x, y : array_like
        x and y coordinates of line points.
        
    Returns
    -------
    
    segments : numpy array of (len(x), 1, 2) form
        Numpy array consists of line segments [x[i],x[i+1]], [y[i],y[i+1]].
    
    See Also
    --------
    
    colorline
    
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
       
    '''
    if z is not None:
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(x, y, z=None,
              c=None, 
              ax=None, 
              cmap='copper', 
              norm=plt.Normalize(0.0, 1.0), 
              linewidth=1, 
              alpha=1.0):

    ''' < Colored line plotter (2D and 3D) >
        Make LineCollection (Line3DCollection) from line segments \
        [x[i],x[i+1]], [y[i],y[i+1]] (, [z[i],z[i+1]]) \
        colored with colors from c array and plot it within ax axis. \
        Uses make_segments.
    
    Parameters
    ----------
    x, y : array_like
        x and y coordinates of line points.

    Optional
    --------
    
    z : array_like
        z coordinate of line points.

    c : scalar or array_like
        Values that will be used to calculate colors for line segments.
    
    ax : matplotlib figure axis
        Figure axis to plot colored line in.
    
    cmap : matplotlib colormap object or string.
        Colormap object or its name. Used to calculate colors using
        values from c array and Normalizer object norm.
        
    norm : matplotlib Normalize object
        Used to normalize values from c array.
    
    linewidth : scalar or array_like
        Linewidth(s) for line segments.
        
    alpha : scalar
        Value used for color blending when plotting.
           
    Returns
    -------
    
    lc : matplotlib LineCollection (Line3DCollection) object
        Can be used for custom plotting or something else.   
    
    See Also
    --------
    
    make_segments
    
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
       
    '''

    # Default colors equally spaced on [0,1]:
    if c is None:
        c = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(c, "__iter__"):  # to check for numerical input -- this is a hack
        c = np.array([c])

    c = np.asarray(c)

    segments = make_segments(x, y, z)

    if z is not None:
        lc = Line3DCollection(segments, array=c, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    else:
        lc = LineCollection(segments, array=c, cmap=cmap, norm=norm,
                            linewidth=linewidth, alpha=alpha)

    if ax:
        ax.add_collection(lc)

    return lc
