# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:24:39 2018

@author: Stanislav Bober

Reused code from here:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections

def make_segments(x, y):
    ''' Create numpy array of line segments from x and y coordinates,
        in the correct format for LineCollection.
       
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
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(x, y, z=None, 
              ax=None, 
              cmap='copper', 
              norm=plt.Normalize(0.0, 1.0), 
              linewidth=1, 
              alpha=1.0):

    ''' < Colored line plotter >
        Make LineCollection from line segments [x[i],x[i+1]], [y[i],y[i+1]] \
        colored with colors from z array and plot it within ax axis. \
        Uses make_segments.
    
    Parameters
    ----------
    x, y : array_like
        x and y coordinates of line points.

    Optional
    --------

    z : scalar or array_like
        Values that will be used to calculate colors for line segments.
    
    ax : matplotlib figure axis
        Figure axis to plot colored line in.
    
    cmap : matplotlib colormap object or string.
        Colormap object or its name. Used to calculate colors using
        values from z array and Normalizer object norm.
        
    norm : matplotlib Normalize object
        Used to normalize values from z array.
    
    linewidth : scalar or array_like
        Linewidth(s) for line segments.
        
    alpha : scalar
        Value used for color blending when plotting.
           
    Returns
    -------
    
    lc : matplotlib LineCollection object
        Can be used for custom plotting or something else.   
    
    See Also
    --------
    
    make_segments
    
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
       
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if ax:
        ax.add_collection(lc)

    return lc

