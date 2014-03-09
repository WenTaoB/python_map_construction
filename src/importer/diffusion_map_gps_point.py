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

    #gps_track.visualize_tracks(tracks, style='k.')
    #return

    window_size = 250
    window_center = (447240, 4424780)
    window_SW = (window_center[0]-window_size, window_center[1]-window_size)
    window_NE = (window_center[0]+window_size, window_center[1]+window_size)

    # Extract GPS points in window
    GPS_points = []
    point_idxs = []
    for track_idx in range(0, len(tracks)):
        track = tracks[track_idx]
        for pt_idx in range(0, len(track.utm)):
            pt = (track.utm[pt_idx][0], track.utm[pt_idx][1])
            if pt[0]>=window_SW[0] and pt[0]<=window_NE[0] and \
                    pt[1]>=window_SW[1] and pt[1]<=window_NE[1]:
                GPS_points.append(pt)
                point_idxs.append((track_idx, pt_idx))

    print "In total %d points"%len(GPS_points)
    GPS_points = np.array(GPS_points)
    n_points = len(GPS_points)

    # Compute similarity matrix
    L = np.zeros((n_points,n_points))
    in_track_bonus = 100
    dist_sigma = 20 # in meters
    for i in range(0, n_points):
        L[i,i] = 1.0
        track_i_idxs = point_idxs[i]
        for j in range(0, n_points):
            if i == j:
                L[i,i] = 1.0
            track_j_idxs = point_idxs[j]
            bonus = 1.0
            if track_i_idxs[0] == track_j_idxs[0]:
                if track_j_idxs[1]-track_i_idxs[1]<=1:
                    bonus *= in_track_bonus
            dist = np.linalg.norm(GPS_points[i]-GPS_points[j]) / bonus
            L[i,j] = np.exp(-1*dist*dist/2.0/dist_sigma/dist_sigma)
            if L[i,j] < 0.01:
                L[i,j] = 0.0

    M = np.array(L, copy=True)
    for i in range(0, L.shape[0]):
        M[i,:] /= sum(M[i,:])

    # Synthetic dataset
    #n_points = 1000
    #synthetic_points = np.zeros((n_points,2))
    #d = 5
    #for i in range(0, int(0.4*n_points)):
    #    theta = np.random.rand()*2*np.pi
    #    r = np.random.rand()
    #    synthetic_points[i,0] = -1.0*d + r*np.cos(theta)
    #    synthetic_points[i,1] = r*np.sin(theta)
    #for i in range(int(0.4*n_points), int(0.8*n_points)):
    #    theta = np.random.rand()*2*np.pi
    #    r = np.random.rand()
    #    synthetic_points[i,0] = d + r*np.cos(theta)
    #    synthetic_points[i,1] = r*np.sin(theta)
    #for i in range(int(0.8*n_points), n_points):
    #    synthetic_points[i,0] = 2*d*(np.random.rand()-0.5)
    #    synthetic_points[i,1] = 0.2*(np.random.rand()-0.5)
    #fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(111, aspect='equal')
    #ax.plot(synthetic_points[:,0], synthetic_points[:,1], '.')
    #plt.show()
    #return
    #L = np.zeros((n_points,n_points))
    #dist_sigma = 2 # in meters
    #for i in range(0, n_points):
    #    L[i,i] = 1.0
    #    for j in range(i+1, n_points):
    #        dist = np.linalg.norm(synthetic_points[i,:]-synthetic_points[j,:])
    #        L[i,j] = np.exp(-1*dist*dist/2.0/dist_sigma/dist_sigma)
    #        if L[i,j] < 0.1:
    #            L[i,j] = 0
    #        L[j,i] = L[i,j]

    #M = np.array(L, copy=True)
    #for i in range(0, L.shape[0]):
    #    M[i,:] /= sum(M[i,:])
    
    # Compute Eigen Vectors of M
    print "Start eigen."

    S = np.array(M, copy=True)
    eigs, v = np.linalg.eig(S)

    print "test = ",np.linalg.norm(np.dot(M, v[:,1])-eigs[1]*v[:,1])

    sorted_idxs = np.argsort(eigs)[::-1]
    s = 1
    index = sorted_idxs[s]
    #test = np.dot(M, v[:,index]) - eigs[index]*v[:,index]
    #print "test norm = ", np.amax(test)

    # Compute New embedding
    k = 200
    t = 10
    Y = np.zeros((n_points, k))
    print "New projection."
    for i in range(0, n_points):
        for s in range(0, k):
            Y[i, s] = (eigs[sorted_idxs[s]]**t) * v[i, sorted_idxs[s]]
    
    #fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(111, aspect='equal')
    #ax.plot(Y[:,0], Y[:,1], 'b.')
    #plt.show() 
    # Clustering
    print "Start clustering."
    kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10) 
    kmeans.fit(Y)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    unique_labels = np.unique(labels)
    print unique_labels

    #db = DBSCAN(eps=0.5, min_samples=10).fit(Y)
    #core_samples = db.core_sample_indices_
    #labels = db.labels_
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print "Estimated number of clusters: %d"%n_clusters_

    #unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0,1, len(unique_labels)))
    
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    for k, col in zip(unique_labels, colors):
        my_members = labels == k
        ax.plot(GPS_points[my_members, 0], GPS_points[my_members, 1],
                '.', color=col, markersize=10)
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
