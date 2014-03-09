#!/usr/bin/env python
import sys
import cPickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx

from osm_for_drawing import *
from osm_parser import *

import const

def main():
    if len(sys.argv) != 2:
        print "Error!\nCorrect usage is:\n\t"
        print "python visualize_osm_around_center.py [osm_test_region_for_draw.gpickle]"
        return
   
    #LOC = (447772, 4424300)
    #R = 500

    #LOC = (446458, 4422150)
    #R = 500

    # San Francisco
    LOC = (551281, 4180430) 
    R = 500

    G = nx.read_gpickle(sys.argv[1])

    osm_for_drawing = OSM_DRAW(G)
    easting, northing = osm_for_drawing.node_list()
    edge_list = osm_for_drawing.edge_list()

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    #for segment in edge_list:
    #    u = segment[1][0] - segment[0][0]
    #    v = segment[1][1] - segment[0][1]
    #    ax.arrow(segment[0][0], segment[0][1], u, v, width=0.5, head_width=5,\
    #                head_length=10, overhang=0.5, **arrow_params)
    edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    ax.add_collection(edge_collection)

    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])

    plt.show()

if __name__ == "__main__":
    sys.exit(main())

