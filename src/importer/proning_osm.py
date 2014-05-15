#!/usr/bin/env python
import sys
import cPickle
from optparse import OptionParser

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Point, Polygon
from descartes.patch import PolygonPatch
import networkx as nx

from osm_parser import *
from osm_for_drawing import *
import const

def main():
    parser = OptionParser()
    parser.add_option("-m", "--osm", dest="osm_data", help="Input open street map data (typically in gpickle format)", metavar="OSM_DATA", type="string")
    parser.add_option("-t", "--track_data", dest="track_data", help="Input GPS tracks", metavar="TRACK_DATA", type="string")
    parser.add_option("-o", "--output_osm", dest="output_osm", help="Output file name (suggested extention: gpickle)", metavar="OUTPUT", type="string")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)
    (options, args) = parser.parse_args()
    
    if not options.osm_data:
        parser.error("Input osm_data not found!")
    if not options.track_data:
        parser.error("Input track_data not found!")
    if not options.output_osm:
        parser.error("Output image not specified!")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    G = nx.read_gpickle(options.osm_data)
    components = nx.weakly_connected_components(G)
    H = G.subgraph(components[0])
    G = H

    tracks = 

    osm_for_drawing = OSM_DRAW(G)
    edge_lists = osm_for_drawing.edge_list()
    line_strings = []
    for edge_list in edge_lists:
        line = LineString(edge_list)
        line_strings.append(line)


    fig = plt.figure(figsize=(10, 10))
    ax = plt.Axes(fig, [0., 0., 1., 1.], aspect='equal')
    ax.set_axis_off()
    fig.add_axes(ax)

    ROAD_WIDTH = 7 # in meters
    for line_string in line_strings:
        polygon = line_string.buffer(ROAD_WIDTH)
        patch = PolygonPatch(polygon, facecolor='k', edgecolor='k')
        ax.add_patch(patch)
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    fig.savefig(options.output_img, dpi=100)
    plt.close()
    
    return

if __name__ == "__main__":
    sys.exit(main())

