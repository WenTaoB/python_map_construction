#!/usr/bin/env python
"""
Created on Mon Oct 29, 2013

@author: ChenChen
"""

import sys
import random
import cPickle
import copy
import time

from scipy import weave
from scipy.weave import converters
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm as CM
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from pykalman import KalmanFilter

import gps_track
import douglas_peucker_track_segmentation
import const

def main():
    with open(sys.argv[1], "r") as fin:
        tracks = cPickle.load(fin)
    print "%d tracks loaded."%len(tracks)

    index = 5
    measurements = []
    track = tracks[index]
    t0 = track.utm[0][2]/1e6
    for pt in track.utm:
        t = pt[2]/1e6 - t0
        measurements.append([pt[0], pt[1]])
    measurements = np.asarray(measurements)
    kf = KalmanFilter(n_dim_obs=2, n_dim_state=2).em(measurements, n_iter=100)
    results = kf.smooth(measurements)[0]
    
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot([pt[0] for pt in results],
            [pt[1] for pt in results],
            'r-', linewidth=2)
    ax.plot([pt[0] for pt in track.utm],
            [pt[1] for pt in track.utm],
            'x-')
    #ax.set_xlim([const.SF_small_RANGE_SW[0], const.SF_small_RANGE_NE[0]])
    #ax.set_ylim([const.SF_small_RANGE_SW[1], const.SF_small_RANGE_NE[1]])
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
