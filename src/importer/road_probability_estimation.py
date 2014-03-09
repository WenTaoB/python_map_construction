#!/usr/bin/env python
"""
Created on Mon Oct 29, 2013

@author: ChenChen
"""

import sys
import cPickle
import copy

import matplotlib.pyplot as plt
from matplotlib import cm as CM
import numpy as np
import skimage.filter
import skimage.feature

import gps_track
import const

def image_array_from_points(tracks, n, bound_box):
    """
        Transform GPS points into a image array with bounding box.
    """

    img = np.zeros((n,n))
    display_img = np.zeros((n,n))

    bound_box_ewidth = bound_box[1][0] - bound_box[0][0]
    bound_box_nwidth = bound_box[1][1] - bound_box[0][1]

    for track in tracks:
        for pt in track.utm:
            nx = np.floor((pt[0] - bound_box[0][0]) / bound_box_nwidth * n)
            ny = np.floor((pt[1] - bound_box[0][1]) / bound_box_nwidth * n)

            if nx < 0:
                nx = 0
            if ny < 0:
                ny = 0
            if nx >= n:
                nx = n-1
            if ny >= n:
                ny = n-1
            
            img[nx, ny] += 1.0
            display_img[nx, ny] = 1
    max_value = np.amax(img)
    img /= max_value
    return img, display_img

def main():
    if len(sys.argv) != 2:
        print "Error! Correct usage is:"
        print "\tpython road_probability_estimation.py [input traj file]"
        return

    tracks = gps_track.load_tracks(sys.argv[1])

    # Rasterization
    N = 5000
    img, display_img = image_array_from_points(tracks, 
                                  N, 
                                  [const.RANGE_SW, const.RANGE_NE])
    
    new_img = skimage.filter.gaussian_filter(img, sigma=3)
    print "new max = ", np.amax(new_img)

    peaks = skimage.feature.peak_local_max(img, min_distance = 20)

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    ax.imshow(display_img.T, cmap='gray')
    ax.plot(peaks[:,0], peaks[:,1], '+r')
    ax.set_xlim([0,N])
    ax.set_ylim([0,N])
    plt.show()

if __name__ == "__main__":
    sys.exit(main())

