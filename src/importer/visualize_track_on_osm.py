#!/usr/bin/env python

import sys
import cPickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import osm_for_drawing
import const

def main():
    if len(sys.argv) != 3:
        print "Error!"
        return

    with open(sys.argv[1], 'rb') as in_file:
        drawing_osm = cPickle.load(in_file)

    with open(sys.argv[2], 'rb') as in_file:
        tracks = cPickle.load(in_file)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    easting, northing = drawing_osm.node_list()
    edge_list = drawing_osm.edge_list()
    #print edge_list
    edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    ax.add_collection(edge_collection)

    for track in tracks:
        ax.plot([pt[0] for pt in track.utm],
                [pt[1] for pt in track.utm],
                'k.')
    #ax.plot([pt[0] for pt in tracks[0].utm],
    #            [pt[1] for pt in tracks[0].utm],
    #            '.-', color='r', markersize=5, linewidth=3)
    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
