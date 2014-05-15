#!/usr/bin/env python
"""
Using a unified grid to rasterize GPS tracks.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.signal import convolve2d
from scipy.interpolate import LSQUnivariateSpline, splrep, splev
from scipy import interpolate
from skimage.morphology import skeletonize

import const

def skeleton_extraction(ordered_points):
    """ Extract skeleton from ordered points
        Args:
            - ordered_points: ndarray
        Return:
            - ndarray
    """
    direction = ordered_points[-1] - ordered_points[0]
    emin = min(ordered_points[:,0])
    emax = max(ordered_points[:,0])
    nmin = min(ordered_points[:,1])
    nmax = max(ordered_points[:,1])

    delta = 5.0 # in meter
    ny = int((emax - emin) / delta) + 1
    nx = int((nmax - nmin) / delta) + 1
    img = np.zeros((nx,ny))

    for pt_idx in range(0, ordered_points.shape[0]):
        pt = ordered_points[pt_idx]
        pt_y = int((pt[0]-emin) / delta)
        pt_x = int((pt[1]-nmin) / delta)
        img[pt_x, pt_y] = 1
        if pt_idx >= 1:
            # Rasterize the line
            last_y = int((ordered_points[pt_idx-1][0]-emin) / delta)
            last_x = int((ordered_points[pt_idx-1][1]-nmin) / delta)
            if abs(pt_x-last_x) + abs(pt_y-last_y) >= 10:
                n_insert = max(pt_x-last_x, pt_y-last_y)
                if n_insert == 0:
                    continue
                delta_x = float(pt_x-last_x) / float(n_insert)
                delta_y = float(pt_y-last_y) / float(n_insert)
                for i in range(1, n_insert+1):
                    x = int(last_x + i*delta_x)
                    y = int(last_y + i*delta_y)
                    if x >= 0 and x <= img.shape[0]-1 and y >= 0 and y <= img.shape[1]-1:
                        img[x,y] = 1
   
    convolve_matrix = np.ones((5,5))
    convolved_img = convolve2d(img, convolve_matrix, mode='same') > 1
    skeleton = skeletonize(convolved_img)

    nonzero_idxs = np.nonzero(skeleton)
    # Transform back to the original
    points = []
    for idx in range(0, len(nonzero_idxs[0])):
        northing = nonzero_idxs[0][idx]*delta + nmin
        easting = nonzero_idxs[1][idx]*delta + emin
        points.append(np.array([easting, northing]))
    points = np.array(points) 
    point_kdtree = spatial.cKDTree(points)
    orig_point_kdtree = spatial.cKDTree(ordered_points)
    # Prone the points
    results = []
    pt = ordered_points[0]
    query_pt = min(5, len(points))
    dist, nb_idxs = point_kdtree.query(pt, query_pt)
    avg_pt = np.mean(points[nb_idxs], axis=0)
    results.append(avg_pt)
    for idx in range(1, ordered_points.shape[0]):
        orig_nb_idxs = orig_point_kdtree.query_ball_point(ordered_points[idx], 10)
        pt = np.mean(ordered_points[orig_nb_idxs], axis=0)
        # Search nearby
        dist, nb_idxs = point_kdtree.query(pt, query_pt)
        avg_pt = np.mean(points[nb_idxs], axis=0)
        last_pt = results[-1]
        vec = avg_pt - last_pt
        if np.linalg.norm(vec) < 10:
            continue
        vec /= np.linalg.norm(vec)
        if len(results) >= 3:
            prev_vec = results[-2] - results[-3]
            prev_vec /= np.linalg.norm(prev_vec)
        else:
            prev_vec = ordered_points[-1] - ordered_points[0]
            prev_vec /= np.linalg.norm(prev_vec)
        
        if np.dot(vec, prev_vec) < np.cos(np.pi/4.0):
            continue
        results.append(avg_pt)

    results = np.array(results)
    #arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    #fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(121, aspect='equal')
    #ax.plot(ordered_points[:,0], ordered_points[:,1], '.-', color='gray')
    #ax.plot(results[:,0], results[:,1], '.-r')

    #ax = fig.add_subplot(122, aspect='equal')
    #ax.imshow(skeleton, cmap=plt.cm.gray)

    ##ax.plot(result[:,0], result[:,1], 'x-', color='r')
    ##ax.arrow(result[0,0], result[0,1],
    ##         result[-1,0]-result[0,0],
    ##         result[-1,1]-result[0,1],
    ##         width=0.01, head_width=0.1, fc='r', ec='r',
    ##         head_length=0.1, overhang=0.5, **arrow_params)
    #plt.show()

    return results

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
