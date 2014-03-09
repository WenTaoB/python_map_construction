#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

#!/usr/bin/env python
"""
Created on Wed Oct 09 14:32:37 2013

@author: ChenChen
"""

import sys
import cPickle
import math
import random

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from descartes import PolygonPatch
from shapely.geometry import Polygon

from skimage.transform import hough_line,hough_line_peaks, probabilistic_hough_line
from skimage.filter import canny
from skimage.morphology import skeletonize

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from itertools import cycle

import const

def main():
    if len(sys.argv) != 2:
        print "ERROR! Correct usage is:"
        print "\tpython test_hough_transform.py [gps_point_collection.dat]"
        return
    
    GRID_SIZE = 500
    results = np.zeros((GRID_SIZE, GRID_SIZE), np.float)
    
    # Load GPS points
    with open(sys.argv[1], "rb") as fin:
        point_collection = cPickle.load(fin)
    
    for pt in point_collection:
        y_ind = math.floor((pt[0] - const.RANGE_SW[0]) / (const.RANGE_NE[0] -const.RANGE_SW[0]) * GRID_SIZE)
        x_ind = math.floor((pt[1] - const.RANGE_NE[1]) / (const.RANGE_SW[1] -const.RANGE_NE[1]) * GRID_SIZE)
        results[x_ind, y_ind] += 1.0
        if results[x_ind, y_ind] >= 64:
            results[x_ind, y_ind] = 63
    results /= np.amax(results)
    
    thresholded_results = np.zeros((GRID_SIZE, GRID_SIZE), np.bool)
    
    THRESHOLD = 0.02
    for i in range(0, GRID_SIZE):
        for j in range(0, GRID_SIZE):
            if results[i,j] >= THRESHOLD:
                thresholded_results[i,j] = 1
            else:
                thresholded_results[i,j] = 0
    box_size = 30
    
#    x_ind = random.randint(box_size, GRID_SIZE-box_size)                
#    y_ind = random.randint(box_size, GRID_SIZE-box_size)
    
    x_ind = 405
    y_ind = 373
    
    test_img = thresholded_results[(x_ind-box_size):(x_ind+box_size),\
                                     (y_ind-box_size):(y_ind+box_size)]
    print test_img.shape                                    
                
    #h, theta, d = hough_line(test_img)
#    fig = plt.figure(figsize=(30,16))
#    ax = fig.add_subplot(121, aspect='equal')
#    ax.imshow(test_img, cmap=plt.cm.gray)
#    
#    ax = fig.add_subplot(122)
#    img = skeletonize(test_img)
#    ax.imshow(img, cmap=plt.cm.gray)
#    plt.show()
#    fig = plt.figure(figsize=(30,16))
#    ax = fig.add_subplot(131, aspect='equal')
#    ax.imshow(test_img, cmap=plt.cm.gray)
#    
#    ax = fig.add_subplot(132)
#    ax.imshow(np.log(1+h),
#               extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
#               cmap=plt.cm.gray, aspect=1/1.5)
#    ax = fig.add_subplot(133)
#    ax.imshow(test_img, cmap=plt.cm.gray)
#    rows, cols = test_img.shape
#    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
#        ax.plot((0, cols), (y0, y1), '-r')
#    ax.set_xlim([0, cols])
#    ax.set_ylim([rows, 0])
#    plt.show()
    
    print "point at: ", x_ind, y_ind
    coords = [(y_ind-box_size, x_ind-box_size), (y_ind+box_size, x_ind-box_size),\
              (y_ind+box_size, x_ind+box_size), (y_ind-box_size, x_ind+box_size), \
              (y_ind-box_size, x_ind-box_size)]
    
    bound_box = Polygon(coords)
    patch = PolygonPatch(bound_box, fc='none', ec='red')
    fig = plt.figure(figsize=(30,16))
    rows, cols = thresholded_results.shape
    ax = fig.add_subplot(121, aspect='equal')
    ax.imshow(thresholded_results, cmap=plt.cm.gray)
    ax.add_patch(patch)
    ax.set_xlim([0, cols])
    ax.set_ylim([rows, 0])
    
    ax = fig.add_subplot(122)
    ax.imshow(test_img, cmap=plt.cm.gray)
    lines = probabilistic_hough_line(test_img, 
                                     threshold=20, 
                                     line_length=20, 
                                     line_gap=10)
    rows, cols = test_img.shape
    for line in lines:
        p0, p1 = line
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), '-r', linewidth=2)
    ax.set_xlim([0, cols])
    ax.set_ylim([rows, 0])
#    
#    # Compute line hough
#    line_hough = []
#    for line in lines:
#        p0, p1 = line
#        vec0 = np.array((float(p1[0]-p0[0]), float(p1[1]-p0[1]), 0.0))
#        v_norm = np.cross(vec0, np.array((0.0,0.0,1.0)))
#        v_norm /= np.linalg.norm(v_norm)
#        angle = abs(np.dot(v_norm, np.array((1.0,0,0))))
#        
#        theta = np.arccos(angle) / 0.5 / math.pi
#        
#        vec1 = np.array((p0[0], p0[1], 0.0))
#        r = abs(np.dot(vec1, v_norm)) / 140.0
#        line_hough.append((theta, r))        
#      
#    X = np.array(line_hough)
#    # Compute DBSCAN
#    db = DBSCAN(eps=0.1, min_samples=1).fit(X)
#    core_samples = db.core_sample_indices_
#    labels = db.labels_
#    
#    # Number of clusters in labels, ignoring noise if present.
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#    
#    print "clusters = ", n_clusters_
#    
#    ax = fig.add_subplot(223, aspect='equal')
#  
#    unique_labels = set(labels)
#    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
#    cluster_center = []
#    for k, col in zip(unique_labels, colors):
#        center_theta = 0.0
#        center_d = 0.0
#        center_count = 0
#        if k == -1:
#            # Black used for noise.
#            col = 'k'
#            markersize = 6
#        class_members = [index[0] for index in np.argwhere(labels == k)]
#        cluster_core_samples = [index for index in core_samples
#                                if labels[index] == k]
#        for index in class_members:            
#            x = X[index]
#            if index in core_samples and k != -1:
#                markersize = 14
#                center_count += 1
#                center_theta += x[0]
#                center_d += x[1]
#            else:
#                markersize = 6
#            ax.plot(x[0], x[1], 'o', markerfacecolor=col,
#                    markeredgecolor='k', markersize=markersize)
#        center_d /= center_count
#        center_d *= 140
#        center_theta /= center_count
#        center_theta *= 0.5*math.pi
#        cluster_center.append((center_theta, center_d))
#        
#    ax.set_xlim([-0.1, 1.1])
#    ax.set_ylim([-0.1, 1.1])
#    
#    ax = fig.add_subplot(224)
#    ax.imshow(test_img, cmap=plt.cm.gray)
#    
#    for angle, dist in cluster_center:
#        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
#        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
#        ax.plot((0, cols), (y0, y1), '-r')
#    rows, cols = test_img.shape
#    ax.set_xlim([0, cols])
#    ax.set_ylim([rows, 0])

    plt.show()
if __name__ == "__main__":
    sys.exit(main())