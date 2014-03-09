#!/usr/bin/env python
import sys
import cPickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import osm_for_drawing
import const

def main():
    if len(sys.argv) != 2:
        print "Error!\nCorrect usage is:\n\t"
        print "python visualize_osm_test_region.py [osm_test_region_for_draw.dat]"
        return
    
    with open(sys.argv[1], 'rb') as in_file:
        drawing_osm = cPickle.load(in_file)

    BBOX_WIDTH = 500
    center = (441144, 4422470)
    BBOX_SW = (center[0]-BBOX_WIDTH, center[1]-BBOX_WIDTH)
    BBOX_NE = (center[0]+BBOX_WIDTH, center[1]+BBOX_WIDTH)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    easting, northing = drawing_osm.node_list()
    edge_list = drawing_osm.edge_list()
    #print edge_list
    edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    ax.add_collection(edge_collection)

    index = 3
    #ax.set_xlim([BBOX_SW[0], BBOX_NE[0]])
    #ax.set_ylim([BBOX_SW[1], BBOX_NE[1]])
    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])

    plt.show()

if __name__ == "__main__":
    sys.exit(main())

