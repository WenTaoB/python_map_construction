#!/usr/bin/env python
"""
    Creat small test osm regions according to bounding boxes.
"""
import sys
import cPickle 

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import const
import osm_for_drawing

def main():
    with open(sys.argv[1], 'rb') as in_file:
        drawing_osm = cPickle.load(in_file)
    
    map_nodes = []
    map_edges = []
    for i in range(0,len(const.BB_SW)):
        map_nodes.append(dict())
        map_edges.append([])

    # Iterate over all nodes in drawing_osm
    for key in drawing_osm.nodes.keys():
        node_easting = drawing_osm.nodes[key][0]
        node_northing = drawing_osm.nodes[key][1]

        for i in range(0, len(const.BB_SW)):
            if node_easting <= const.BB_NE[i][0] and node_easting >= const.BB_SW[i][0]:
                if node_northing <= const.BB_NE[i][1] and node_northing >= const.BB_SW[i][1]:
                    map_nodes[i][key] = (node_easting, node_northing)

    # Iterate over all edges in drawing_osm
    for edge in drawing_osm.edges:
        for i in range(0, len(const.BB_SW)):
            is_start_in = False
            is_end_in = False
            if map_nodes[i].has_key(edge[0]):
                is_start_in = True
            if map_nodes[i].has_key(edge[1]):
                is_end_in = True

            if is_start_in or is_end_in:
                # Add edge into edge list
                map_edges[i].append(edge)
                if not is_start_in:
                    map_nodes[i][edge[0]] = drawing_osm.nodes[edge[0]]
                if not is_end_in:
                    map_nodes[i][edge[1]] = drawing_osm.nodes[edge[1]]

    # Create OSM_DRAW instances
    output_filename_prefix = sys.argv[2]
    for i in range(0, len(const.BB_SW)):
        output_filename = output_filename_prefix + "_%d"%i + ".dat"
        test_osm = osm_for_drawing.OSM_DRAW(map_nodes[i], map_edges[i])
        with open(output_filename, 'wb') as out_file:
            cPickle.dump(test_osm, out_file, protocol=2)

if __name__ == "__main__":
    sys.exit(main())
