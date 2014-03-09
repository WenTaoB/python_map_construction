#!/usr/bin/env python
"""
Created on Sun Oct 06 22:02:23 2013

@author: ChenChen
"""

import sys
import cPickle
import math
import random

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM

import const

def main():
    if len(sys.argv) != 2:
        print "ERROR! Correct usage is:"
        print "\tpython unified_grid.py [gps_point_collection.dat]"
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
    loc_i = 300
    loc_j = 100
    delta = 10
    
    feature_collection = []
    feature_loc = []
    MAX_COUNT = -1
    MAX_SING_VAL = -1.0    
    
    index = 1
    while index <= 100:
        feature = []
        loc_i = random.randint(10, GRID_SIZE-10)
        loc_j = random.randint(10, GRID_SIZE-10)
        feature_loc.append((loc_i, loc_j))
        data_array = thresholded_results[(loc_i-delta):(loc_i+delta+1), (loc_j-delta):(loc_j+delta+1)]
        
        # First order feature
        tmp_feature = []
        for r in range(0, delta+1):
            count = 0
            for i in range(loc_i-r, loc_i+r+1):
                for j in range(loc_j-r, loc_j+r+1):
                    if thresholded_results[i,j]:
                        count += 1
            tmp_feature.append(float(count))
        count_feature = np.array(tmp_feature)
        max_count = max(count_feature)
        if max_count > MAX_COUNT:
            MAX_COUNT = max_count
        
        U,s,V = np.linalg.svd(data_array)
        max_sing_val = max(s)
        if max_sing_val > MAX_SING_VAL:
            MAX_SING_VAL = max_sing_val

        feature.extend(tmp_feature)        
        feature.extend(s)
        feature_collection.append(feature)
        fig = plt.figure(figsize=(32,16))
        ax = fig.add_subplot(131, aspect='equal')
        ax.imshow(thresholded_results, cmap=CM.gray_r)
        ax.plot(loc_j, loc_i, 'xr', markersize=16)
        ax.set_xlim([0, GRID_SIZE])
        ax.set_ylim([0, GRID_SIZE])
        
        ax = fig.add_subplot(132)
        ax.plot(range(0,delta+1), count_feature, '.-')
        
        ax = fig.add_subplot(133)
        ax.plot(s, '.-')
        out_filename = "figures\\eval\\"+"%d"%index + ".png"
        index += 1
        plt.savefig(out_filename)
        plt.close(fig)

    # Feature scaling
    for feature in feature_collection:
        for ind in range(0, delta+1):
            feature[ind] /= MAX_COUNT
        for ind in range(delta+1, len(feature)):
            feature[ind] /= MAX_SING_VAL
    
    with open("eval_feature_loc.dat", "wb") as fout:
        cPickle.dump(feature_loc, fout, protocol=2)
    with open("eval_feature_collection.dat", "wb") as fout:
        cPickle.dump(feature_collection, fout, protocol=2)

if __name__ == "__main__":
    sys.exit(main())