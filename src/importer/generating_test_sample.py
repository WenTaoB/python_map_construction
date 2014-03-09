#!/usr/bin/env python
"""
Created on Mon Oct 07 17:19:49 2013

@author: ChenChen
"""

import sys
import math
import random
import cPickle
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from shapely.geometry import LineString
from scipy import spatial

import const
import gps_track

def extract_GPS_points_in_region(tracks, center, box_size):
    """
        Extract GPS points from a test region.
        Args:
            - tracks: GPS tracks;
            - center: a tuple, (center_x, center_y);
            - box_size: the length of the edge of the bounding box, in meters.
        Return:
            - a list of GPS points:
                [(p1_e, p1_n), (p2_e, p2_n), ...]
    """

    e_min = center[0] - box_size*0.5
    e_max = center[0] + box_size*0.5
    n_min = center[1] - box_size*0.5
    n_max = center[1] + box_size*0.5

    point_collection = []
    for track in tracks:
        for pt in track.utm:
            if pt[0] <= e_max and pt[0] >= e_min and\
                    pt[1] <= n_max and pt[1] >= n_min:
                point_collection.append(pt)

    data = np.array([[pt[0] for pt in point_collection],\
                     [pt[1] for pt in point_collection]]).T

    return data

def extract_GPS_point_in_region_with_direction(tracks,
                                               center,
                                               box_size):
    """
        Extract GPS points from a test region.
        Args:
            - tracks: GPS tracks;
            - center: a tuple, (center_x, center_y);
            - box_size: the length of the edge of the bounding box, in meters.
        Return:
            - a list of GPS points:
                [(p1_e, p1_n), (p2_e, p2_n), ...]
    """
    e_min = center[0] - box_size*0.5
    e_max = center[0] + box_size*0.5
    n_min = center[1] - box_size*0.5
    n_max = center[1] + box_size*0.5

    point_collection = []
    for track in tracks:
        for p_idx in range(1, len(track.utm)-1):
            pt = track.utm[p_idx]
            nxt_pt = track.utm[p_idx+1]
            if pt[0] <= e_max and pt[0] >= e_min and\
                    pt[1] <= n_max and pt[1] >= n_min:
                # Compute direction
                direction = np.array([nxt_pt[0]-pt[0], nxt_pt[1]-pt[1]])
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 10.0:
                    direction *= 0.0
                else:
                    direction /= direction_norm
                point_collection.append((pt[0], pt[1], direction[0], direction[1]))

    data = np.array([[pt[0] for pt in point_collection],\
                     [pt[1] for pt in point_collection],\
                     [pt[2] for pt in point_collection],\
                     [pt[3] for pt in point_collection]]).T
    
    return data

def main():
    tracks = gps_track.load_tracks(sys.argv[1])
  
    bound_box = [(446057, 4423750), (447057, 4424750)]
    gps_track.visualize_tracks(tracks, bound_box = bound_box, style='.')

    return

    #CENTER_PT = (447820, 4423040) # example 1
    #CENTER_PT = (446557, 4424250) #example 2
    #CENTER_PT = (447379, 4422790) #example 3
    #CENTER_PT = (449765, 4424340) #example 4

    BOX_SIZE = 1000

    # Without direction
    #point_collection = extract_GPS_points_in_region(tracks, CENTER_PT, BOX_SIZE)
   
    # With direction
    data = extract_GPS_point_in_region_with_direction(tracks, CENTER_PT, BOX_SIZE)

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(data[:,0], data[:,1], '.', color='gray')
    plt.show()

    with open("test_data/point_collection_with_direction/example_4.dat", "w") as fout:
        cPickle.dump(data, fout, protocol=2)

    return

if __name__ == "__main__":
    sys.exit(main())
