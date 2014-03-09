#!/usr/bin/env python
"""
Created on Wed Oct 09 20:21:30 2013

@author: ChenChen
"""

import sys
import cPickle
import math
import random

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM

from skimage.segmentation import felzenszwalb, slic, quickshift, join_segmentations
from skimage.segmentation import mark_boundaries

import const

def main():
    if len(sys.argv) != 2:
        print "ERROR! Correct usage is:"
        print "\tpython test_random_walker.py [gps_point_collection.dat]"
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
                
    #segments_fz = felzenszwalb(results, scale=100, sigma=0.5, min_size=50)
    segments_slic = join_segmentations(results, ratio=10, n_segments=250, sigma=1)
    #segments_quick = quickshift(results, kernel_size=3, max_dist=6, ratio=0.5)
                
    fig = plt.figure(figsize=(30,16))
    ax = fig.add_subplot(121, aspect='equal')
    ax.imshow(results, cmap=plt.cm.gray)
    
    ax = fig.add_subplot(122)
    ax.imshow(mark_boundaries(results, segments_slic))

    plt.show()
if __name__ == "__main__":
    sys.exit(main())