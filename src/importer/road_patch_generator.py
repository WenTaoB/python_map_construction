#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import sys
import cPickle
import time
import copy
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import networkx as nx

import gps_track
from point_cloud import PointCloud
from road_segment import RoadSegment

import const

class Edge:
    def __init__(self):
        pass

def main():
    parser = OptionParser()
    parser.add_option("-s", "--sample_point_cloud", dest="sample_point_cloud", help="Input sample point cloud filename", metavar="SAMPLE_POINT_CLOUD", type="string")
    parser.add_option("-r", "--road_segment", dest="road_segment", help="Input road segment filename", metavar="ROAD_SEGMENT", type="string")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)
    (options, args) = parser.parse_args()
    
    if not options.sample_point_cloud:
        parser.error("Input sample_point_cloud filename not found!")
    if not options.road_segment:
        parser.error("Input road segment file not found!")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    with open(options.sample_point_cloud, 'rb') as fin:
        sample_point_cloud = cPickle.load(fin)

    with open(options.road_segment, 'rb') as fin:
        road_segments = cPickle.load(fin)

    # Assign each sample point a set of road segments that might contain it
    sample_covers = []
    THRESHOLD = 50 # in meters
    for i in range(0, sample_point_cloud.locations.shape[0]):
        sample_cover = []
        for j in range(0, len(road_segments)):
            segment = road_segments[j]
            road_start = segment.center - segment.half_length*segment.direction
            road_end = segment.center + segment.half_length*segment.direction
            vec1 = sample_point_cloud.locations[i] - road_start
            vec2 = sample_point_cloud.locations[i] - road_end
            if np.dot(vec1, vec2) < 0:
                dist = abs(np.dot(vec1, segment.norm_dir))
                if dist < THRESHOLD:
                    sample_cover.append(j)
        sample_covers.append(sample_cover)
    print "Computing sample cover done."

    sample_kdtree = spatial.cKDTree(sample_point_cloud.locations)
    for i in range(0, len(road_segments)):
        segment = road_segments[i]
        search_radius = 2*segment.half_width
        n_pt_interpolation = max(3, int(4*segment.half_length / search_radius))
        road_start = segment.center - segment.half_length*segment.direction
        road_end = segment.center + segment.half_length*segment.direction
        pt_xs = np.linspace(road_start[0], road_end[0], n_pt_interpolation)
        pt_ys = np.linspace(road_start[1], road_end[1], n_pt_interpolation)
        nearby_point_idxs = []
        for j in range(0, n_pt_interpolation):
            nb_idxs = sample_kdtree.query_ball_point(np.array([pt_xs[j],pt_ys[j]]), search_radius)
            nearby_point_idxs.extend(nb_idxs)
        nearby_point_idxs = list(set(nearby_point_idxs))

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot(sample_point_cloud.locations[:,0],
           sample_point_cloud.locations[:,1],
           '.', color='gray')
    ax.plot([road_start[0], road_end[0]],
           [road_start[1], road_end[1]],'r-')
    ax.plot(sample_point_cloud.locations[nearby_point_idxs,0],
            sample_point_cloud.locations[nearby_point_idxs,1],'r.')
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show()

    return
    
if __name__ == "__main__":
    sys.exit(main())
