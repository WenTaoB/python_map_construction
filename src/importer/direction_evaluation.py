#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import sys
import cPickle
import math
import random
import time

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.collections import LineCollection
from scipy import spatial
import networkx as nx

from osm_parser import *
from osm_for_drawing import *
import const

def main():
    GRID_SIZE = 2.5 # in meters
    # Target location and radius
    # test_point_cloud.dat
    #LOC = (447772, 4424300)
    #R = 500

    # test_point_cloud1.dat
    #LOC = (446458, 4422150)
    #R = 500

    # San Francisco
    LOC = (551281, 4180430) 
    R = 500

    if len(sys.argv) != 4:
        print "ERROR! Correct usage is:"
        print "\tpython sample_hog_feature.py [sample_point_cloud.dat] [sample_direction.dat] [osm.gpickle]"
        return

    with open(sys.argv[1], 'rb') as fin:
        sample_point_cloud = cPickle.load(fin)

    with open(sys.argv[2], 'rb') as fin:
        sample_directions = cPickle.load(fin)
  
    G = nx.read_gpickle(sys.argv[3]) 
    components = nx.weakly_connected_components(G)
    H = G.subgraph(components[0])
    G = H

    osm_for_drawing = OSM_DRAW(G)
    easting, northing = osm_for_drawing.node_list()
    edge_list = osm_for_drawing.edge_list()
    edge_directions = []
    edge_norms = []
    for edge in edge_list:
        start = np.array(edge[0])
        end = np.array(edge[1])
        direction = end - start
        direction /= np.linalg.norm(direction)
        edge_directions.append(direction)
        edge_norms.append(np.array([-1*direction[1], direction[0]]))
  
    incorrect_sample_idxs = []
    angle_threshold = np.cos(np.pi/6.0)
    for sample_idx in range(0, sample_point_cloud.directions.shape[0]):
        min_dist = np.inf
        min_dist_idx = -1
        for edge_idx in range(0, len(edge_list)):
            vec1 = sample_point_cloud.locations[sample_idx] - np.array(edge_list[edge_idx][0])
            vec2 = sample_point_cloud.locations[sample_idx] - np.array(edge_list[edge_idx][1])
            if np.dot(vec1, vec2) < 0:
                dist = abs(np.dot(vec1, edge_norms[edge_idx]))
                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx = edge_idx
            else:
                continue
        # Check direction
        if min_dist > 25:
            continue
        sample_correct = False
        for sample_dir in sample_directions[sample_idx]:
            if np.dot(sample_dir, edge_directions[min_dist_idx]) > angle_threshold:
                sample_correct = True
                break
        if not sample_correct:
            incorrect_sample_idxs.append(sample_idx)

    correctness = 100 - float(len(incorrect_sample_idxs)) / sample_point_cloud.directions.shape[0] * 100
    print "Correctness is %.2f %%"%correctness
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    ax.add_collection(edge_collection)

    ax.plot(sample_point_cloud.locations[incorrect_sample_idxs,0],
            sample_point_cloud.locations[incorrect_sample_idxs,1],
            'or')

    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    ##ax = fig.add_subplot(122)
    ##ax.plot(np.arange(n_bin), hog_feature, '.-')
    ##ax.set_xlim([0,n_bin+1])
    ##ax.set_ylim([-0.1,1.1])
    plt.show()    
    return

if __name__ == "__main__":
    sys.exit(main())
