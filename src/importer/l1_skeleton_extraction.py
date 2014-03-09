#!/usr/bin/env python
"""
Using a unified grid to rasterize GPS tracks.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.interpolate import LSQUnivariateSpline, splrep, splev
from scipy import interpolate

import const

def skeleton_extraction(ordered_points):
    """ Extract skeleton from ordered points
        Args:
            - ordered_points: ndarray
        Return:
            - ndarray
    """
    direction = ordered_points[-1] - ordered_points[0]
    principal_dir = direction / np.linalg.norm(direction)

    mean_pt = np.mean(ordered_points, axis=0)
    
    mean_pts = np.copy(ordered_points) - mean_pt
    
    M_rot = np.array([[principal_dir[0], principal_dir[1]], [-1*principal_dir[1], principal_dir[0]]])
    M_inverse_rot = np.array([[principal_dir[0], -1*principal_dir[1]], [principal_dir[1], principal_dir[0]]])
    rotated_data = np.dot(M_rot, mean_pts.T).T

    sorted_idxs = np.argsort(rotated_data[:,0])
    sorted_data = rotated_data[sorted_idxs]
    xs = np.linspace(sorted_data[0,0], sorted_data[-1,0], 10)
    s = interpolate.interp1d(sorted_data[:,0], 
                             sorted_data[:,1])
    ys = s(xs)
    rotated_result = np.array((xs, ys)).T
    result = np.dot(M_inverse_rot, rotated_result.T).T + mean_pt

    #arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    #fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(111)
    #ax.plot(sorted_data[:,0], sorted_data[:,1], '.-', color='gray')
    #ax.plot(xs, ys, '.-r')
    ##ax.plot(result[:,0], result[:,1], 'x-', color='r')
    ##ax.arrow(result[0,0], result[0,1],
    ##         result[-1,0]-result[0,0],
    ##         result[-1,1]-result[0,1],
    ##         width=0.01, head_width=0.1, fc='r', ec='r',
    ##         head_length=0.1, overhang=0.5, **arrow_params)
    #plt.show()

    return result

def main():
    x = np.linspace(0,1,100)
    y = np.sin(4*x) + 0.5*np.random.rand(len(x))
    points = np.array((x,y)).T
    result = skeleton_extraction(points)
    
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    fig = plt.figure(figsize=const.figsize)
    ax.arrow(result[0,0], result[0,1],
             result[-1,0]-result[0,0],
             result[-1,1]-result[0,1],
             width=0.01, head_width=0.1, fc='r', ec='r',
             head_length=0.1, overhang=0.5, **arrow_params)
    #         result[-1,1]-result[0,1],
    #         width=0.01, head_width=0.1, fc='r', ec='r',
    #         head_length=0.1, overhang=0.5, **arrow_params)
    plt.show()
    return

if __name__ == "__main__":
    sys.exit(main())
