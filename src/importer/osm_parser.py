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
import pyproj
import numpy as np
from shapely.geometry import LineString
from descartes import PolygonPatch

import const

def draw_osm(G, ax):
    """
        Plot the osm graph on the figure defined by axis ax.
        Args:
            - G: networkx openstreetmap graph
            - ax: matplotlib axis, ax = fig.add_subplot()
        Return:
            - No return
    """
    # Plot nodes
    easting = []
    northing = []
    for node in G.nodes():
        easting.append(G.node[node]['data'][0])
        northing.append(G.node[node]['data'][1])
    min_easting = min(easting)
    max_easting = max(easting)
    center_easting = 0.5*min_easting + 0.5*max_easting
    min_northing = min(northing)
    max_northing = max(northing)
    center_northing = 0.5*min_northing + 0.5*max_northing
    R = 0.55*max(max_easting-min_easting, max_northing-min_northing)

    ax.plot(easting, northing, 'sb', markersize=12)

    # Plot edges
    count = 0
    for edge in G.edges_iter():
        color = const.colors[count%7]
        linestring = G[edge[0]][edge[1]]['linestring']
        tags = G[edge[0]][edge[1]]['data'].tags
        start_pt = np.array(G.node[edge[0]]['data'])
        end_pt = np.array(G.node[edge[1]]['data'])

        if tags.has_key('oneway'):
            if tags['oneway'] == 'yes':
                vec = end_pt - start_pt
                ax.arrow(start_pt[0],
                         start_pt[1],
                         vec[0],
                         vec[1],
                         width=2, head_width=10, fc=color, ec=color,
                         head_length=20, overhang=0.5, **const.arrow_params)
        else:
            ax.plot([start_pt[0], end_pt[0]],
                    [start_pt[1], end_pt[1]], '-',
                    color=color, linewidth=2)
        count += 1
    
    ax.set_xlim([center_easting-R, center_easting+R])
    ax.set_ylim([center_northing-R, center_northing+R])

def visualize_osm_as_patches(G, with_width=False):
    """ Return a list of descartes polygonpatches for map visualization
        Args:
            - G: osm as networkx graph
        Return:
            - patches: a list of PolygonPatch objects, the patches can be added using ax.
                       add_patch(patches)
    """
    patches = []

    road_width = {'motorway': 1, 
                  'trunk': 1, 
                  'primary': 1, 
                  'secondary': 1, 
                  'tertiary': 1,
                  'unclassified': 1,
                  'residential': 1,
                  'service': 1,
                  'motorway_link': 1,
                  'trunk_link': 1,
                  'primary_link': 1,
                  'secondary_link': 1,
                  'tertiary_link': 1}
    if with_width:
        road_width = {'motorway': 20, 
                      'trunk': 15, 
                      'primary': 10, 
                      'secondary': 8, 
                      'tertiary': 8,
                      'unclassified': 8,
                      'residential': 6,
                      'service': 6,
                      'motorway_link': 8,
                      'trunk_link': 6,
                      'primary_link': 5,
                      'secondary_link': 5,
                      'tertiary_link': 5}

    road_order = {'motorway': 5, 
                  'trunk': 5, 
                  'primary': 4, 
                  'secondary': 4, 
                  'tertiary': 4,
                  'unclassified': 3,
                  'residential': 3,
                  'service': 2,
                  'motorway_link': 5,
                  'trunk_link': 5,
                  'primary_link': 4,
                  'secondary_link': 4,
                  'tertiary_link': 4}

    road_color = {'motorway': '#0000ff', 
                  'trunk': '#5ca028', 
                  'primary': '#cc3333', 
                  'secondary': '#ffcb05', 
                  'tertiary': 'y',
                  'unclassified': '#808080',
                  'residential': '#000000',
                  'service': '#000000',
                  'motorway_link': '#0000ff',
                  'trunk_link': '#5ca028',
                  'primary_link': '#cc3333',
                  'secondary_link': '#ffcb05',
                  'tertiary_link': 'y'}

    for edge in G.edges_iter():
        linestring = G[edge[0]][edge[1]]['linestring']
        tags = G[edge[0]][edge[1]]['data'].tags

        order = 0
        if road_width.has_key(tags['highway']):
            width = road_width[tags['highway']]
            color = road_color[tags['highway']]
            order = road_order[tags['highway']]
        else:
            width = 2
            color = '#800080'
            print tags['highway']
        dilated = linestring.buffer(width) 
        patch = PolygonPatch(dilated, fc=color, ec=color, alpha=0.7, zorder=order)
        patches.append(patch)

    return patches

def read_osm(filename, only_roads=True):
    """ Read graph in OSM format from file specified by name

        Returns
            G: graph

        Examples:
            G = read_osm("beijing.osm")
    """
    osm = OSM(filename)
    G = networkx.DiGraph()
    ways = []

    things_to_remove = {'footway':1, 'bridleway':1, 'steps':1,'pedestrian':1, 'construction':1,
            'proposed':1, 'bus_stop':1,
            'path':1, 'cycleway':1, 'raceway':1, 'track':1, 'road':1, 'living_street':1,
            'services':1, 'no':1, 'platform':1}

    for w in osm.ways.itervalues():
        if only_roads and 'highway' not in w.tags:
            continue
        if things_to_remove.has_key(w.tags['highway']):
            continue

        way_locs = []
        for nd in w.nds:
            loc = osm.nodes[nd]
            way_locs.append((loc.easting, loc.northing))

        linestring = LineString(way_locs)
        G.add_edge(w.nds[0], w.nds[-1], id=w.id, linestring=linestring, data=w)

    for n_id in G.nodes_iter():
        n = osm.nodes[n_id]
        G.node[n_id]['data'] = (n.easting, n.northing)

    return G

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
