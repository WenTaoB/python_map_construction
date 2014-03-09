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
import ransac

import const

def main():
    if len(sys.argv) != 2:
        print "ERROR! Correct usage is:"
        print "\tpython test_ransac.py [gps_point_collection.dat]"
        return

    with open(sys.argv[1], "rb") as fin:
        point_collection = cPickle.load(fin)
    
    # Setup model
    model = ransac.LinearLeastSquaresModel([0,1],[2])
    
    point_ind = random.randint(0, len(point_collection))    
    box_size = 100
    
    nearby_point = []
    
    for pt in point_collection:
        if pt[0] >= point_collection[point_ind][0] - box_size\
           and pt[0] <= point_collection[point_ind][0] + box_size\
           and pt[1] >= point_collection[point_ind][1] - box_size\
           and pt[1] <= point_collection[point_ind][1] + box_size:
               nearby_point.append((pt[0], pt[1], -100000))
    all_data = np.array(nearby_point)
    
    print all_data.shape
    d = math.floor(0.2*len(nearby_point))
    ransac_fit, ransac_data = ransac.ransac(all_data, 
                                            model, d, 1000, 100, d, 
                                            return_all=True)
    sort_idxs = np.argsort(all_data[:,0])
    data_sorted = all_data[sort_idxs]
    
    fit_y = (-100000 - ransac_fit[0]*data_sorted[:,0])/ransac_fit[1]
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot([pt[0] for pt in point_collection],
            [pt[1] for pt in point_collection],
            '.', color='gray')
    ax.plot([pt[0] for pt in nearby_point],
            [pt[1] for pt in nearby_point],
            '.')
    ax.plot(data_sorted[:,0],
            fit_y, 'r-')
    plt.show()
if __name__ == "__main__":
    sys.exit(main())