#!/usr/bin/env python
import sys
import cPickle
import copy

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import networkx as nx

from osm_parser import *
from osm_for_drawing import *
import const

def edge_to_points(start_pt, end_pt, sampling_density):
    points = []
    directions = []

    vec = np.array(end_pt) - np.array(start_pt)
    vec_norm = np.linalg.norm(vec)
    direction = vec / vec_norm
    print vec_norm
    n_point_to_insert = int(vec_norm / sampling_density)
    print n_point_to_insert
    delta_v = vec / (n_point_to_insert+1)
    print delta_v

    points.append(start_pt)
    directions.append(direction)
    for i in range(0, n_point_to_insert):
        pt = points[-1] + delta_v
        points.append(pt)
        directions.append(direction)
    points.append(end_pt)
    directions.append(direction)
    print np.array(points)
    return np.array(points), np.array(directions)

class MapDecomposition:
    def __init__(self):
        pass

    def primitive_decomposition(self, G, sampling_density):
        """ Using linesegments, L-segment and arc to decompose the map G.
        """
        nodes = G.nodes()
        marked_edges = {}
        for node in nodes:
            node = nodes[29]
            node_successors = G.successors(node)
            node_predecessors = G.predecessors(node)
            valid_in_edges = [] 
            valid_out_edges = []
            for successor in node_successors:
                if not marked_edges.has_key((node, successor)):
                    valid_in_edges.append((node, successor))
            if len(valid_in_edges) == 0:
                continue
            for predecessor in node_predecessors:
                if not marked_edges.has_key((predecessor, node)):
                    valid_out_edges.append((predecessor, node))
            if len(valid_out_edges) == 0:
                continue
            
            osm_for_drawing = OSM_DRAW(G)
            easting, northing = osm_for_drawing.node_list()
            edge_list = osm_for_drawing.edge_list()
            fig = plt.figure(figsize=const.figsize)
            ax = fig.add_subplot(111, aspect='equal')
            #print edge_list
            #for segment in edge_list:
            #    u = segment[1][0] - segment[0][0]
            #    v = segment[1][1] - segment[0][1]
            #    ax.arrow(segment[0][0], segment[0][1], u, v, width=0.5, head_width=5,\
            #                head_length=10, overhang=0.5, **arrow_params)
            edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
            ax.add_collection(edge_collection)
            ax.plot(G.node[node]['data'].easting, G.node[node]['data'].northing,
                    'or')            

            best_fit_n = 0
            for successor in node_successors:
                for predecessor in node_predecessors:
                    starting_point = (G.node[predecessor]['data'].easting,\
                                      G.node[predecessor]['data'].northing)
                    ending_point = (G.node[node]['data'].easting,\
                                    G.node[node]['data'].northing)
                    points_1, directions_1 = edge_to_points(starting_point, 
                                                            ending_point, 
                                                            sampling_density)
                    starting_point = (G.node[node]['data'].easting,\
                                      G.node[node]['data'].northing)
                    ending_point = (G.node[successor]['data'].easting,\
                                    G.node[successor]['data'].northing)
                    points_2, directions_2 = edge_to_points(starting_point, 
                                                            ending_point, 
                                                            sampling_density)
                    points = np.concatenate((points_1, points_2))
                    directions = np.concatenate((directions_1, directions_2))

                    # fit line

                    # fit L-shape

                    # fit circle

                     
            arrow_params = {'fc':'r', 'ec':'r', 'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
            for sucessor in node_successors:
                x = G.node[node]['data'].easting
                y = G.node[node]['data'].northing
                u = G.node[successor]['data'].easting - x
                v = G.node[successor]['data'].northing - y
                ax.arrow(x, y, u, v, width=0.5, head_width=5,\
                         head_length=10, overhang=0.5, **arrow_params)

            for predecessor in node_predecessors:
                x = G.node[predecessor]['data'].easting
                y = G.node[predecessor]['data'].northing
                u = G.node[node]['data'].easting - x
                v = G.node[node]['data'].northing - y
                ax.arrow(x, y, u, v, width=0.5, head_width=5,\
                         head_length=10, overhang=0.5, **arrow_params)
            ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
            ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])

            plt.show() 
            sys.exit(1)
def main():
    if len(sys.argv) != 2:
        print "Error!\nCorrect usage is:\n\t"
        print "python visualize_osm_test_region.py [osm_test_region_for_draw.dat]"
        return
   
    G = nx.read_gpickle(sys.argv[1])

    components = nx.weakly_connected_components(G)
    print "There are %d connected components."%len(components)

    H = G.subgraph(components[0])
    G = H

    osm_for_drawing = OSM_DRAW(G)
    easting, northing = osm_for_drawing.node_list()
    edge_list = osm_for_drawing.edge_list()

    #map_decomposition = MapDecomposition()
    #map_decomposition.primitive_decomposition(G, 10.0)

    #return

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    #print edge_list
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    for segment in edge_list:
        u = segment[1][0] - segment[0][0]
        v = segment[1][1] - segment[0][1]
        ax.arrow(segment[0][0], segment[0][1], u, v, width=0.5, head_width=5,\
                    head_length=10, overhang=0.5, **arrow_params)
    #edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    #ax.add_collection(edge_collection)

    # Junction nodes
    for node in G.nodes():
        if G.degree(node) > 2:
            ax.plot(G.node[node]['data'].easting,
                    G.node[node]['data'].northing,
                    'ro')

    # Connected components
    #for index in range(0, len(components)):
    #    color = const.colors[index%7]
    #    print len(components[index])
    #    for node in components[index]:
    #        ax.plot(G.node[node]['data'].easting,
    #                G.node[node]['data'].northing,
    #                'o', color=color)
    #    break

    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])

    plt.show()

if __name__ == "__main__":
    sys.exit(main())
