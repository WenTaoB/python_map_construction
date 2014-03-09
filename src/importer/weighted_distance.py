#!/usr/bin/env python
"""
Using a unified grid to rasterize GPS tracks.
"""

import sys
import cPickle

from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from skimage.feature import peak_local_max, corner_peaks, hog
from skimage.transform import hough_line,probabilistic_hough_line
from skimage.filter import gaussian_filter

import networkx as nx

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from itertools import cycle

import gps_track
import const

def main():
    tracks = gps_track.load_tracks(sys.argv[1])

    window_size = 250.0
    window_center = (447217, 4424780)
    window_SW = (window_center[0]-window_size, window_center[1]-window_size)
    window_NE = (window_center[0]+window_size, window_center[1]+window_size)
    
    #gps_track.visualize_tracks(tracks, bound_box=[window_SW, window_NE], style='k.')
    gps_track.visualize_tracks(tracks, style='k.')
    return

    # Extract GPS points in window
    GPS_points = []
    point_idxs = []
    for track_idx in range(0, len(tracks)):
        track = tracks[track_idx]
        for pt_idx in range(0, len(track.utm)):
            pt = (track.utm[pt_idx][0], track.utm[pt_idx][1])
            if pt[0]>=window_SW[0] and pt[0]<=window_NE[0] and \
                    pt[1]>=window_SW[1] and pt[1]<=window_NE[1]:
                new_pt = (pt[0]-window_center[0], pt[1]-window_center[1])
                GPS_points.append(new_pt)
                point_idxs.append((track_idx, pt_idx))
    print "In total %d points"%len(GPS_points)
    GPS_points = np.array(GPS_points)

    t = 5.0
    h = 10.0
    sigma = 10.0
    deformed_points = []
    for pt in GPS_points:
        r_sum = pt[0]**2 + pt[1]**2
        ratio = np.exp(-1.0*r_sum/2.0/sigma/sigma)
        new_e = pt[0]*(1 + t*ratio)
        new_n = pt[1]*(1 + t*ratio)
        new_z = h*ratio
        deformed_points.append((new_e, new_n, new_z))
    deformed_points = np.array(deformed_points)

    N = 100
    x = np.linspace(-window_size, window_size, N)
    y = np.linspace(-window_size, window_size, N)
    #xx, yy = np.meshgrid(x, y)
    xx = np.zeros((N,N))
    yy = np.zeros((N,N))
    zz = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(N):
            r_sum = x[i]**2 + y[j]**2
            ratio = np.exp(-1*r_sum/2/sigma/sigma)
            xx[i,j] = x[i]*(1 + t*ratio)
            yy[i,j] = y[j]*(1 + t*ratio)
            zz[i,j] = h*ratio - 0.3

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(xx, yy, zz, color='gray')
    ax.scatter(deformed_points[:,0], deformed_points[:,1], deformed_points[:,2], 'r') 
    plt.show()
    return
    
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, projection='3d')
    N = 50
    x = np.linspace(-5.0, 5.0, N)
    y = np.linspace(-5.0, 5.0, N)
    #xx, yy = np.meshgrid(x, y)
    xx = np.zeros((N,N))
    yy = np.zeros((N,N))
    zz = np.zeros((N,N))
    sigma = 1.0
    h = 5.0
    t = 5.0
    for i in np.arange(N):
        for j in np.arange(N):
            r_sum = x[i]**2 + y[j]**2
            ratio = np.exp(-1*r_sum/2/sigma/sigma)
            xx[i,j] = x[i]*(1 + t*ratio)
            yy[i,j] = y[j]*(1 + t*ratio)
            zz[i,j] = h*ratio
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.jet,
        linewidth=0, antialiased=False)
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
