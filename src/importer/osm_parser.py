"""
    Read openstreetmap osm format, and produce a networkx graph.
    Created by Chen Chen on 01-22-2014.
    Referenced by codes from: https://gist.github.com/aflaxman/287370/
"""
import sys
import cPickle

import xml.sax
import copy
import networkx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pyproj
import numpy as np

import osm_for_drawing

def read_osm(filename, only_roads=True):
    """ Read graph in OSM format from file specified by name

        Returns
            G: graph

        Examples:
            G = read_osm("beijing.osm")
    """
    osm = OSM(filename)
    G = networkx.Graph()
    ways = []

    for w in osm.ways.itervalues():
        if only_roads and 'highway' not in w.tags:
            continue
        G.add_path(w.nds, id=w.id, data=w)
        ways.append(w.nds)
    for n_id in G.nodes_iter():
        n = osm.nodes[n_id]
        G.node[n_id] = dict(data=n)
    return G, ways

class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        utm_projector = pyproj.Proj(proj='utm', zone=50, south=False, ellps='WGS84')
        easting, northing = utm_projector(lon, lat)
        self.easting = easting
        self.northing = northing
        self.tags = {}
       
class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}

    def split(self, dividers):
        # Slice the node-array using this nifty recursive function
        def slice_array(ar, dividers):
            for i in range(1, len(ar)-1):
                if dividers[ar[i]]>1:
                    left = ar[:i+1]
                    right = ar[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left]+rightsliced
            return [ar]
        
        slices = slice_array(self.nds, dividers)

        # Create a way object for each node-aray slice
        ret = []
        i = 0
        for slice in slices:
            littleway = copy.copy( self )
            littleway.id += "-%d"%i
            littleway.nds = slice
            ret.append( littleway )
            i += 1
        return ret

class OSM:
    def __init__(self, filename_or_stream):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}
        
        superself = self
        
        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self,loc):
                pass
            
            @classmethod
            def startDocument(self):
                pass
                
            @classmethod
            def endDocument(self):
                pass
                
            @classmethod
            def startElement(self, name, attrs):
                if name=='node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name=='way':
                    self.currElem = Way(attrs['id'], superself)
                elif name=='tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name=='nd':
                    self.currElem.nds.append( attrs['ref'] )
                
            @classmethod
            def endElement(self,name):
                if name=='node':
                    nodes[self.currElem.id] = self.currElem
                elif name=='way':
                    ways[self.currElem.id] = self.currElem
                
            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)
        self.nodes = nodes
        self.ways = ways
        #count times each node is used
        node_histogram = dict.fromkeys( self.nodes.keys(), 0 )
        for way in self.ways.values():
            if len(way.nds) < 2:       #if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1
        
        #use that histogram to split all ways, replacing the member set of ways
        new_ways = {}
        for id, way in self.ways.iteritems():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways
