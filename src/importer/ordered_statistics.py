#!/usr/bin/env python
"""
Created on Mon Oct 07 17:19:49 2013

@author: ChenChen
"""

import sys
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM

import const
import gps_track


def point_to_ij(pt,
                bound_box,
                grid_size):
    """
        Convert UTM point location to index in a matrix
    """
    i = math.floor(float(pt[0] - bound_box[0][0])*grid_size/float(bound_box[1][0]-bound_box[0][0]))
    j = math.floor(float(pt[1] - bound_box[0][1])*grid_size/float(bound_box[1][1]-bound_box[0][1]))
    return (int(i),int(j))

def ij_to_id(index, matrix_size):
    """
        Convert index in a matrix to a number.
            - index: (i,j)
            - matrix_size: a shape, (nx,ny)
            Formula:
                id = i*n_j + j
        Return:
            - number as id.
    """
    return int(index[0]*matrix_size[1] + index[1])

def main():
    if len(sys.argv) != 2:
        print "Error! Correct usage is:"
        print "\tpython ordered_statistics.py [input_tracks]"
        return
    tracks = gps_track.load_tracks(sys.argv[1])

    GRID_SIZE = 1000

    MAX_ORDER = 6
    TIME_STEP = 30

    # Zero-th order
    rastered_samples = np.zeros((GRID_SIZE, GRID_SIZE))
    for track in tracks:
        for pt in track.utm:
            (i,j) = point_to_ij(pt,
                                (const.RANGE_SW, const.RANGE_NE),
                                GRID_SIZE)
            rastered_samples[i,j] += 1
    display_rastered_samples = np.log10(rastered_samples)
    feature = []
    for order in range(1, MAX_ORDER+1):
        print "Now in order ", order
        TIME_THRES_HIGH = TIME_STEP * order
        TIME_THRES_LOW = TIME_STEP * (order - 1)
        this_order_feature = {}
        for track in tracks:
            for pt_ind in range(0, len(track.utm)):
                nxt_pt_ind = pt_ind + 1
                while nxt_pt_ind < len(track.utm):
                    # Check time diff
                    time_diff = track.utm[nxt_pt_ind][2] - track.utm[pt_ind][2]
                    if time_diff <= TIME_THRES_HIGH:
                        if time_diff >= TIME_THRES_LOW:
                            (from_i, from_j) = point_to_ij(track.utm[pt_ind],
                                               (const.RANGE_SW, const.RANGE_NE),
                                                GRID_SIZE)
                            (to_i, to_j) = point_to_ij(track.utm[nxt_pt_ind],
                                           (const.RANGE_SW, const.RANGE_NE),
                                                GRID_SIZE)
                            key = "%d,%d,%d,%d"%(from_i, from_j, to_i, to_j)
                            if this_order_feature.has_key(key):
                                this_order_feature[key] += 1
                            else:
                                this_order_feature[key] = 1
                    else:
                        break
                    nxt_pt_ind += 1
        feature.append(this_order_feature)

    print "Calculation completed."

    # Pick a point
    WINDOW_SIZE = 40

    while True:
        ind_i = random.randint(WINDOW_SIZE, GRID_SIZE-WINDOW_SIZE)
        ind_j = random.randint(WINDOW_SIZE, GRID_SIZE-WINDOW_SIZE)
        if rastered_samples[ind_i, ind_j] >= 20:
            break

    point_feature = []

    key_prefix = "%d,%d,"%(ind_i, ind_j)
    key_postfix = ",%d,%d"%(ind_i, ind_j)

    for order in range(0, MAX_ORDER):
        f = feature[order]
        this_point_feature = np.zeros((2*WINDOW_SIZE+1, 2*WINDOW_SIZE+1))
        for key in f.keys():
            if key.startswith(key_prefix):
                ind_str = key.split(',')
                to_i = int(ind_str[2]) - ind_i + WINDOW_SIZE
                to_j = int(ind_str[3]) - ind_j + WINDOW_SIZE
                if to_i == WINDOW_SIZE and to_j == WINDOW_SIZE:
                    continue
                if to_i >=0 and to_i < 2*WINDOW_SIZE+1 and\
                    to_j >= 0 and to_j < 2*WINDOW_SIZE+1:
                    this_point_feature[to_i, to_j] += 1
            if key.endswith(key_postfix):
                ind_str = key.split(',')
                from_i = int(ind_str[2]) - ind_i + WINDOW_SIZE
                from_j = int(ind_str[3]) - ind_j + WINDOW_SIZE
                if from_i == WINDOW_SIZE and from_j == WINDOW_SIZE:
                    continue
                if from_i >=0 and from_i < 2*WINDOW_SIZE+1 and\
                    from_j >= 0 and from_j < 2*WINDOW_SIZE+1:
                    this_point_feature[from_i, from_j] += 1
        point_feature.append(this_point_feature)

    # Visualization
   # fig = plt.figure()
   # ax = fig.add_subplot(231, aspect='equal')
   # ax.imshow(display_rastered_samples)
   # ax.plot(ind_j, ind_i, 'r+', markersize=12)
   # ax.set_xlim([0,GRID_SIZE])
   # ax.set_ylim([0,GRID_SIZE])
#  #  ax.set_xlim([ind_j-WINDOW_SIZE,ind_j+WINDOW_SIZE])
#  #  ax.set_ylim([ind_i-WINDOW_SIZE,ind_i+WINDOW_SIZE])
   #
   # ax = fig.add_subplot(232, aspect='equal')
   # ax.imshow(point_feature[0])
   # ax.set_xlim([0,2*WINDOW_SIZE])
   # ax.set_ylim([0,2*WINDOW_SIZE])
   #
   # ax = fig.add_subplot(233, aspect='equal')
   # ax.imshow(point_feature[1])
   # ax.set_xlim([0,2*WINDOW_SIZE])
   # ax.set_ylim([0,2*WINDOW_SIZE])
   #
   # ax = fig.add_subplot(234, aspect='equal')
   # ax.imshow(point_feature[2])
   # ax.set_xlim([0,2*WINDOW_SIZE])
   # ax.set_ylim([0,2*WINDOW_SIZE])
   #
   # ax = fig.add_subplot(235, aspect='equal')
   # ax.imshow(point_feature[3])
   # ax.set_xlim([0,2*WINDOW_SIZE])
   # ax.set_ylim([0,2*WINDOW_SIZE])
   #
   # ax = fig.add_subplot(236, aspect='equal')
   # ax.imshow(point_feature[4])
   # ax.set_xlim([0,2*WINDOW_SIZE])
   # ax.set_ylim([0,2*WINDOW_SIZE])

    # Visualization
    fig = plt.figure(figsize=(32,16))
    ax = fig.add_subplot(231, aspect='equal')
    ax.imshow(display_rastered_samples)
    ax.set_xlim([0,GRID_SIZE])
    ax.set_ylim([0,GRID_SIZE])

    PLOT_THRES = 2

    ax = fig.add_subplot(232, aspect='equal')
    count = 0
    x_ind = []
    y_ind = []
    for key in feature[0]:
        if feature[0][key] < PLOT_THRES:
            continue
        count += 1
#        if count == 1000:
#            break
        ind_str = key.split(',')
        from_i = int(ind_str[0])
        from_j = int(ind_str[1])
        to_i = int(ind_str[2])
        to_j = int(ind_str[3])
        x_ind.append(from_i)
        x_ind.append(to_i)
        x_ind.append(None)
        y_ind.append(from_j)
        y_ind.append(to_j)
        y_ind.append(None)
    ax.plot(y_ind, x_ind, 'r-')
    ax.set_xlim([0,GRID_SIZE])
    ax.set_ylim([0,GRID_SIZE])
    print "order 0, count = ",count

    ax = fig.add_subplot(233, aspect='equal')
    ax.imshow(rastered_samples, cmap=CM.gray_r)
    x_ind = []
    y_ind = []
    count = 0
    for key in feature[1]:
        if feature[1][key] < PLOT_THRES:
            continue
        count += 1
        ind_str = key.split(',')
        from_i = int(ind_str[0])
        from_j = int(ind_str[1])
        to_i = int(ind_str[2])
        to_j = int(ind_str[3])
        x_ind.append(from_i)
        x_ind.append(to_i)
        x_ind.append(None)
        y_ind.append(from_j)
        y_ind.append(to_j)
        y_ind.append(None)
    ax.plot(y_ind, x_ind, 'r-')
    ax.set_xlim([0,GRID_SIZE])
    ax.set_ylim([0,GRID_SIZE])
    print "order 1, count = ",count

    ax = fig.add_subplot(234, aspect='equal')
    ax.imshow(rastered_samples, cmap=CM.gray_r)
    x_ind = []
    y_ind = []
    count = 0
    for key in feature[2]:
        if feature[2][key] < PLOT_THRES:
            continue
        count += 1
        ind_str = key.split(',')
        from_i = int(ind_str[0])
        from_j = int(ind_str[1])
        to_i = int(ind_str[2])
        to_j = int(ind_str[3])
        x_ind.append(from_i)
        x_ind.append(to_i)
        x_ind.append(None)
        y_ind.append(from_j)
        y_ind.append(to_j)
        y_ind.append(None)
    ax.plot(y_ind, x_ind, 'r-')
    ax.set_xlim([0,GRID_SIZE])
    ax.set_ylim([0,GRID_SIZE])
    print "order 2, count = ",count

    ax = fig.add_subplot(235, aspect='equal')
    ax.imshow(rastered_samples, cmap=CM.gray_r)
    x_ind = []
    y_ind = []
    count = 0
    for key in feature[3]:
        if feature[3][key] < PLOT_THRES:
            continue
        count += 1
        ind_str = key.split(',')
        from_i = int(ind_str[0])
        from_j = int(ind_str[1])
        to_i = int(ind_str[2])
        to_j = int(ind_str[3])
        x_ind.append(from_i)
        x_ind.append(to_i)
        x_ind.append(None)
        y_ind.append(from_j)
        y_ind.append(to_j)
        y_ind.append(None)
    ax.plot(y_ind, x_ind, 'r-')
    ax.set_xlim([0,GRID_SIZE])
    ax.set_ylim([0,GRID_SIZE])
    print "order 3, count = ",count

    ax = fig.add_subplot(236, aspect='equal')
    ax.imshow(rastered_samples, cmap=CM.gray_r)
    x_ind = []
    y_ind = []
    count = 0
    for key in feature[4]:
        if feature[4][key] < PLOT_THRES:
            continue
        count += 1
        ind_str = key.split(',')
        from_i = int(ind_str[0])
        from_j = int(ind_str[1])
        to_i = int(ind_str[2])
        to_j = int(ind_str[3])
        x_ind.append(from_i)
        x_ind.append(to_i)
        x_ind.append(None)
        y_ind.append(from_j)
        y_ind.append(to_j)
        y_ind.append(None)
    ax.plot(y_ind, x_ind, 'r-')
    ax.set_xlim([0,GRID_SIZE])
    ax.set_ylim([0,GRID_SIZE])
    print "order 4, count = ",count

    plt.show()

if __name__ == "__main__":
    sys.exit(main())
