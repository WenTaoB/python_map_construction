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

import primitive_fitting

class MapDecomposition:
    def __init__(self):
        pass

    def straight_line_decomposition(self, G, error_threshold):
        """ Use straight line segments to decompose the map.
            Args:
                - G: the directional graph;
                - error_threshold: AVERAGE error per point!
            Return:
                - segments: of format [[(start_e, start_n), (end_e, end_n)]].
        """
        segments = []
        marked_edges = {}
        line_model = primitive_fitting.LinearCurveModel()
        count = 0

        #fig = plt.figure(figsize=const.figsize) 
        #ax = fig.add_subplot(111, aspect="equal")
        
        for node in G.nodes():
            for current_node in G.successors(node):
                edge = (node, current_node)
                if marked_edges.has_key(edge):
                    continue
                else:
                    marked_edges[edge] = 1

                path_points = []
                path_nodes = []
                segment_start = np.array((G.node[node]['data'].easting, G.node[node]['data'].northing))
                path_points.append(segment_start)
                path_nodes.append(node)
                
                current_model = np.array([0.0, 0.0, 0.0])

                segment_end = np.array((G.node[current_node]['data'].easting, G.node[current_node]['data'].northing))
                path_points.append(segment_end)
            
                while True:
                    # Search through successors
                    is_extended = False
                    for successor in G.successors(current_node):
                        edge = (current_node, successor)
                        if marked_edges.has_key(edge):
                            continue

                        tmp_end = np.array((G.node[successor]['data'].easting, G.node[successor]['data'].northing))
                        vec1 = np.array((path_points[-1][0]-path_points[0][0], path_points[-1][1]-path_points[0][1]))
                        vec2 = np.array((tmp_end[0]-path_points[-1][0], tmp_end[1]-path_points[-1][1]))
                        # check compatibility
                        if np.dot(vec1, vec2) < 0:
                            continue

                        # Try to fit a line
                        tmp_path_points = copy.deepcopy(path_points)
                        tmp_path_points.append(tmp_end)
                        data = np.array(tmp_path_points)
                        model = line_model.fit(data) 
                        err_per_point = line_model.get_error(data, model)
                        if err_per_point > error_threshold:
                            continue
                        else:
                            path_points.append(tmp_end)
                            path_nodes.append(successor)
                            marked_edges[edge] = 1
                            is_extended = True
                            current_model = model
                            current_node = successor
                            break
                    if not is_extended:
                        break
                
                count += 1
                print "count=",count
                if len(path_points) == 2:
                    segment = path_points
                else:
                    path_points = np.array(path_points)
                    segment = line_model.get_segment(path_points, current_model)
                    length = np.linalg.norm(np.array((segment[1][0]-segment[0][0], segment[1][1]-segment[0][1])))
                path_points = np.array(path_points)
                #ax.plot(path_points[:,0], path_points[:,1], '-', color='gray')
                segments.append(segment)

        #ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
        #ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])

        #plt.show()
        return segments

def main():
    if len(sys.argv) != 2:
        print "Error!\nCorrect usage is:\n\t"
        print "python visualize_osm_test_region.py [osm_test_region_for_draw.gpickle]"
        return
   
    G = nx.read_gpickle(sys.argv[1])

    osm_for_drawing = OSM_DRAW(G)
    easting, northing = osm_for_drawing.node_list()
    edge_list = osm_for_drawing.edge_list()

    map_decomposition = MapDecomposition()
    segments = map_decomposition.straight_line_decomposition(G, 5.0)

    lengths = []
    for segment in segments:
        vec = np.array([segment[1][0]-segment[0][0], segment[1][1]-segment[0][1]])
        lengths.append(np.linalg.norm(vec))
    
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    ax.hist(lengths, 100) 
    ax.set_xlabel('Length (meter)', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)
    plt.show() 
    return

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    #print edge_list
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    #for segment in edge_list:
    #    u = segment[1][0] - segment[0][0]
    #    v = segment[1][1] - segment[0][1]
    #    ax.arrow(segment[0][0], segment[0][1], u, v, width=0.5, head_width=5,\
    #                head_length=10, overhang=0.5, **arrow_params)
    edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    ax.add_collection(edge_collection)

    for segment in segments:
        ax.plot([segment[0][0], segment[1][0]],
                [segment[0][1], segment[1][1]],
                '-r', linewidth=2)

    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])

    plt.show()

if __name__ == "__main__":
    sys.exit(main())
