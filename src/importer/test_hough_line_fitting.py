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
    box_size = 50
    
    x_ind = random.randint(box_size, GRID_SIZE-box_size)                
    y_ind = random.randint(box_size, GRID_SIZE-box_size)
    
#    x_ind = 217
#    y_ind = 175

    
    test_img = thresholded_results[(x_ind-box_size):(x_ind+box_size),\
                                     (y_ind-box_size):(y_ind+box_size)]
                
    h, theta, d = hough_line(test_img)
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
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=8)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        ax.plot((0, cols), (y0, y1), '-r')
    rows, cols = test_img.shape
    ax.set_xlim([0, cols])
    ax.set_ylim([rows, 0])
    plt.show()
if __name__ == "__main__":
    sys.exit(main())