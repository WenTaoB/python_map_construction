#!/usr/bin/env python
"""
Created by Chen Chen on 01-22-2014.
"""
from shapely.geometry import LineString, Point, Polygon
import osm_parser

class OSM_DRAW:
    def __init__(self, G):
        """
            G: osm graph
        """
        self.G = G
        self.nodes = G.nodes()
        self.edges = G.edges()

    def node_list(self):
        easting = []
        northing = []
        for node in self.nodes:
            easting.append(self.G.node[node]['data'].easting)
            northing.append(self.G.node[node]['data'].northing)
        return easting, northing

    def edge_list(self, colors='r', linewidths=2):
        """
            Return LineCollection containing all edges.
        """
        lines = []
        for val in self.edges:
            start_e = self.G.node[val[0]]['data'].easting
            start_n = self.G.node[val[0]]['data'].northing
            end_e = self.G.node[val[1]]['data'].easting
            end_n = self.G.node[val[1]]['data'].northing
            lines.append([(start_e, start_n), (end_e, end_n)])
        return lines 
    
    def build_path(self, path):
        path_easting = []
        path_northing = []
        for node in path:
            path_easting.append(self.G.node[node]['data'].easting)
            path_northing.append(self.G.node[node]['data'].northing)
        return path_easting, path_northing
