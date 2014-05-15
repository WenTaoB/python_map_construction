#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import sys
import cPickle
import math
import random
import time
import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.collections import LineCollection
from scipy import spatial
from scipy import signal

from skimage.transform import hough_line,hough_line_peaks, probabilistic_hough_line
from skimage.filter import canny
from skimage.morphology import skeletonize

from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import gps_track
from point_cloud import PointCloud
import l1_skeleton_extraction

import const

def main():
       
    compute_canonical_dir = False
    
    GRID_SIZE = 2.5 # in meters

    # Target location and radius
    # test_point_cloud.dat
    LOC = (447772, 4424300)
    R = 500

    # test_point_cloud1.dat
    #LOC = (446458, 4422150)
    #R = 500

    if len(sys.argv) != 3:
        print "ERROR! Correct usage is:"
        print "\tpython sample_hog_feature.py [sample_point_cloud.dat] [sample_direction.dat]"
        return

    with open(sys.argv[1], 'rb') as fin:
        sample_point_cloud = cPickle.load(fin)

    with open(sys.argv[2], 'rb') as fin:
        sample_directions = cPickle.load(fin)
    
    sample_point_kdtree = spatial.cKDTree(sample_point_cloud.locations)
    mean_pt = np.mean(sample_point_cloud.locations, axis=0)
    K = 10
    n_bin = 16
    delta_angle = 360.0 / n_bin
    alpha = 0.0
    
    sample_features = []
    for sample_idx in range(0, sample_point_cloud.locations.shape[0]):
        dist, nb_idxs = sample_point_kdtree.query(sample_point_cloud.locations[sample_idx], K)
        loc_feature = sample_point_cloud.locations[sample_idx] - mean_pt
        loc_feature = loc_feature / 1000 * alpha
        hog_feature = np.array([0.0]*n_bin)
        for pt_idx in nb_idxs:
            for direction in sample_directions[pt_idx]:
                angle = np.arccos(np.dot(direction, np.array([1.0, 0.0])))
                angle = np.rad2deg(angle)
                if direction[1] < 0:
                    angle = 360 - angle
                angle_bin = int(angle / delta_angle) 
                hog_feature[angle_bin] += 1
        sum_feature = sum(hog_feature) 
        if sum_feature > 1e-3:
            hog_feature /= sum(hog_feature)
        sample_feature = np.concatenate((loc_feature, hog_feature), axis=0)
        sample_features.append(sample_feature)
        
    sample_features = np.array(sample_features)
    print "There are %d samples."%len(sample_features)

    # DBSCAN clustering
    db = DBSCAN(eps=0.05, min_samples=5).fit(sample_features)
    core_samples = db.core_sample_indices_
    labels = db.labels_

    # number of clusters, ignoring noise if present
    unique_labels = set(labels)
    
    ## MeanShift clustering
    #bandwidth = estimate_bandwidth(sample_features, quantile=0.2, n_samples=500)
    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #ms.fit(sample_features)
    #labels = ms.labels_
    #unique_labels = set(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "There are %d clusters"%n_clusters_

    horizontal_road = np.array([0.0]*(n_bin+2))
    horizontal_road[0] = 0.25
    horizontal_road[1] = 0.25
    horizontal_road[-1] = 0.25
    horizontal_road[-2] = 0.25
    
    #distances = np.array([0.0]*sample_point_cloud.locations.shape[0])
    #for sample_idx in range(0, sample_point_cloud.locations.shape[0]):
    #    vec = sample_features[sample_idx] - horizontal_road
    #    distances[sample_idx] = np.linalg.norm(vec)
    #print "min distance: ",min(distances)
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    for k in unique_labels:
        markersize = 12
        if k == -1:
            col = 'k'
            markersize = 6
        color = const.colors[np.random.randint(7)]
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_sample_locations = sample_point_cloud.locations[class_members]
        ax.plot(cluster_sample_locations[:,0],
                cluster_sample_locations[:,1],
                '.', color=color)
    
    #ax.plot(sample_point_cloud.locations[:,0],
    #        sample_point_cloud.locations[:,1],
    #        '.', color='gray')
    #error = 0.8
    #ax.plot(sample_point_cloud.locations[distances<error,0],
    #        sample_point_cloud.locations[distances<error,1],
    #        'r.')
    #ax.plot(sample_point_cloud.locations[sample_idx,0],
    #        sample_point_cloud.locations[sample_idx,1],
    #        'or')
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    #ax = fig.add_subplot(122)
    #ax.plot(np.arange(n_bin), hog_feature, '.-')
    #ax.set_xlim([0,n_bin+1])
    #ax.set_ylim([-0.1,1.1])
    plt.show()    
    return

if __name__ == "__main__":
    sys.exit(main())
