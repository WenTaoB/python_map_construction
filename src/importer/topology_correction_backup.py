#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import sys
import cPickle
import time
import copy
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from descartes.patch import PolygonPatch
from cvxpy import *

import gps_track
from point_cloud import PointCloud
from road_segment import RoadSegment

import const

class TrackLink:
    def __init__(self):
        self.track_id = -1
        self.start_pt = np.zeros(2)
        self.end_pt = np.zeros(2)
        self.delta_t = 0.0

class Edge:
    def __init__(self):
        self.source_segment = -1
        self.end_segment = -1
        self.tracklink_list = []

class RoadPatch:
    def __init__(self):
        self.center_line = np.array([])
        self.road_widths = np.array([])
        self.direction_computed = False

    def compute_segment_dir(self):
        directions = []
        norm_dir = []
        for i in range(0, len(self.center_line)-1):
            vec = self.center_line[i+1] - self.center_line[i]
            vec_length = np.linalg.norm(vec)
            if vec_length < 1.0:
                directions.append(np.array([0.0, 0.0]))
                norm_dir.append(np.array([0.0, 0.0]))
                continue

            vec /= vec_length
            vec_norm = np.array([-1*vec[1], vec[0]])
            directions.append(vec)
            norm_dir.append(vec_norm)
        self.directions = np.array(directions)
        self.norm_directions = np.array(norm_dir)
        self.direction_computed = True

    def road_polygon(self):
        if not self.direction_computed:
            self.compute_segment_dir()
        center_line_string = LineString(self.center_line)
        polygon = center_line_string.buffer(self.road_widths[0])
        return polygon

def track_simplification(track,
                         start_idx,
                         end_idx,
                         d_threshold = 10.0):
    """
        Simplify a track so that its cleaner, and direction at each points can be infered more easily.
        
        Args:
            - track: an object of GpsTrack class
            - start_idx: starting point index, in track.utm
            - end_idx: ending point index, in track.utm
            - d_threshold: cutting threshold for Douglas Peuker algorithm
        Return:
            - simplified_track: a list of points - [pt_0, pt_1, ...]
    """
    if start_idx >= end_idx-1:
        return []

    pt_vec = []
    for i in range(start_idx, end_idx):
        pt_vec.append((track.utm[i][0], track.utm[i][1]))
    pt_vec = np.array(pt_vec)

    vec_start_end = pt_vec[-1] - pt_vec[0]
    vec_length = np.linalg.norm(vec_start_end)
    if vec_length < 0.01:
        return []
    vec_start_end /= vec_length
    vec_norm = np.array((-1*vec_start_end[1], vec_start_end[0]))

    delta_pt = pt_vec - pt_vec[0]

    dist_to_line = np.abs(np.dot(delta_pt, vec_norm))
    max_dist_idx = np.argmax(dist_to_line)
    max_pt_idx = max_dist_idx + start_idx

    if dist_to_line[max_dist_idx] > d_threshold:
        part1 = dp_segmentation_idx(track, start_idx, max_pt_idx+1, d_threshold)
        part2 = dp_segmentation_idx(track, max_pt_idx, end_idx, d_threshold)

        result = []
        result.extend(list(part1))
        result.extend(list(part2))
        return result

    return [start_idx, end_idx]

def road_segment_graph(tracks, 
                       road_segments):
    """
        Generate road segment graph by traversing each trajectory.
        Args:
            - tracks: a list of tracks
            - road_segments: a list of road segments
        return:
            - G:
                - Nodes: road_segment_id
                - Edge: object of Edge class
    """
    G = nx.Graph()
    segment_idxs = np.arange(len(road_segments))
    G.add_nodes_from(segment_idxs)

    # Traversing through each track
    for t in tracks:
        simplified_track = track_simplification(t)
        for pt_idx in range(0, len(t.utm)):
            if pt_idx == 0:
                pass
            elif pt_idx == len(t.utm) - 1:
                pass
            else:
                pass
    
    return G

def segment_graph_to_map(G,
                         road_segments,
                         sample_point_cloud):
    """ Merging segment graph G to produce a map.
            Args:
                - G: a directed networkx graph for road segments
                - road_segments: a list of RoadSegment objects
                - sample_point_cloud
                - sample_idx_on_roads
            Return:
                - G_map: a map as a directed networkx graph
    """
    road_patches = []
    pass


def project_tracks_to_road(tracks, 
                           road_segments):
    """
        Compute tracks that fall into each road segment.
            Args:
                - tracks
                - road_segments
            Return:
                - track_on_road: a dictionary, each key is the index of a road segment. Each 
                                 value is a set of indices of the tracks fall onto this road.
    """
    track_on_road = {}
    for seg_idx in np.arange(len(road_segments)):
        track_on_road[seg_idx] = set([])

    simplified_tracks = []
    for track in tracks:
        line = LineString([(pt[0], pt[1]) for pt in track.utm])
        simplified_track = line.simplify(10.0)
        simplified_tracks.append(simplified_track)
   
    # Compute road segment linestrings
    road_segment_linestrings = []
    for r_seg in road_segments:
        r_start = r_seg.center - r_seg.half_length*r_seg.direction
        r_end = r_seg.center + r_seg.half_length*r_seg.direction
        r_linestring = LineString([r_start, r_end])
        road_segment_linestrings.append(r_linestring)

    for seg_idx in np.arange(len(road_segments)):
        print seg_idx
        for track_idx in np.arange(len(tracks)):
            if road_segment_linestrings[seg_idx].distance(simplified_tracks[track_idx]) > 1.2*road_segments[seg_idx].half_width:
                continue
            track = tracks[track_idx]
            if len(track.utm) <= 1:
                continue
            for utm_idx in np.arange(len(track.utm)):
                utm = track.utm[utm_idx]
                if utm_idx == 0:
                    direction = np.array([track.utm[utm_idx+1][0], track.utm[utm_idx+1][1]]) - \
                                np.array([track.utm[utm_idx][0], track.utm[utm_idx][1]])
                elif utm_idx == len(track.utm) - 1:
                    direction = np.array([track.utm[utm_idx][0], track.utm[utm_idx][1]]) - \
                                np.array([track.utm[utm_idx-1][0], track.utm[utm_idx-1][1]])
                else:
                    direction = np.array([track.utm[utm_idx+1][0], track.utm[utm_idx+1][1]]) - \
                                np.array([track.utm[utm_idx-1][0], track.utm[utm_idx-1][1]])

                direction /= np.linalg.norm(direction)
                if np.dot(direction, road_segments[seg_idx].direction) < np.cos(np.pi/4.0):
                    continue

                pt = Point(utm[0], utm[1])
                if pt.distance(road_segment_linestrings[seg_idx]) < 1.2*road_segments[seg_idx].half_width:
                    track_on_road[seg_idx].add(track_idx)
                    break

    return track_on_road

def track_to_segment_projction(track,
                               road_segments,
                               angle_threshold = np.pi/4.0):
    """ Project tracks to road segments
        Args:
            - track: a GPS track
            - road_segments: a list of RoadSegment
            - dist_threshold: in meters
        Return:
            - related_segments: an index list
    """
    road_segment_linestrings = []
    for r_seg in road_segments:
        r_start = r_seg.center - r_seg.half_length*r_seg.direction
        r_end = r_seg.center + r_seg.half_length*r_seg.direction
        r_linestring = LineString([r_start, r_end])
        road_segment_linestrings.append(r_linestring)

    related_segments = []
    line = LineString([(pt[0], pt[1]) for pt in track.utm])
    simplified_line = line.simplify(10.0)
    coords = simplified_line.coords
    for pt_idx in range(1, len(coords)):
        track_segment_dir = np.array(coords[pt_idx]) - np.array(coords[pt_idx-1])
        track_segment_dir_norm = np.linalg.norm(track_segment_dir)
        if track_segment_dir_norm < 1.0:
            continue
        track_segment_dir /= np.linalg.norm(track_segment_dir)
        track_segment = LineString([coords[pt_idx-1], coords[pt_idx]])
        for r_seg_idx in np.arange(len(road_segments)):
            r_segment = road_segments[r_seg_idx]
            if np.dot(track_segment_dir, r_segment.direction) >= np.cos(angle_threshold):
                dist = track_segment.distance(road_segment_linestrings[r_seg_idx])
                if dist <= r_segment.half_width:
                    related_segments.append(r_seg_idx)

    nearby_segments = list(set(related_segments))

    if len(nearby_segments) <= 1:
        return nearby_segments

    # Select through nearby segments to get the cover of the trajctory
    m = len(track.utm)
    n = len(nearby_segments)
    A = np.zeros((m,n))

    for pt_idx in np.arange(len(track.utm)):
        point = Point(track.utm[pt_idx][0], track.utm[pt_idx][1])
        for nb_seg_idx in np.arange(len(nearby_segments)):
            seg_idx = nearby_segments[nb_seg_idx]
            if point.distance(road_segment_linestrings[seg_idx]) <= road_segments[seg_idx].half_width:
                A[pt_idx, nb_seg_idx] = 1.0
    
    alpha = 0.5
    x = Variable(n)
    delta = Variable(m)
    objective = Minimize(sum(alpha*x)+sum(delta))
    constraints = [0 <= x, 
                   x <= 1,
                   0 <= delta,
                   delta <= 1,
                   A*x >= 1 - delta]
    prob = Problem(objective, constraints)
    prob.solve()
    selected_segments = []
    order = []
    for i in np.arange(len(x.value)):
        if x.value[i] > 0.5:
            selected_segments.append(nearby_segments[i])
            order_of_this_segment = np.average(np.where(A[:, i])[0])
            order.append(order_of_this_segment)
    order = np.array(order)
    sorted_idxs = np.argsort(order)

    sorted_segments = []
    for idx in sorted_idxs:
        sorted_segments.append(selected_segments[idx])

    return sorted_segments

def generate_road_patch_from_track(track,
                                   road_segments,
                                   sample_point_cloud,
                                   sample_idx_on_roads,
                                   dist_threshold = 25.0):
    """ Generate road patch from track
        Args:
            - track: GpsTrack object
            - road_segments
            - sample_point_cloud
            - sample_idx_on_roads
            - dist_threshold: in meters
        Return:
            - road_patches: a list of RoadPatch objects
    """
    line = LineString([(pt[0], pt[1]) for pt in track.utm])
    simplified_track = line.simplify(10.0)

    nearby_seg_idxs = track_to_segment_projction(track, road_segments)
    nearby_sample_idxs = set([])
    for seg_idx in nearby_seg_idxs:
        for sample_idx in sample_idx_on_roads[seg_idx]:
            pt = Point(sample_point_cloud.locations[sample_idx][0],
                       sample_point_cloud.locations[sample_idx][1])
            if pt.distance(simplified_track) < dist_threshold:
                nearby_sample_idxs.add(sample_idx)
    nearby_sample_idxs = np.array(list(nearby_sample_idxs))
    if len(nearby_sample_idxs) == 0:
        return []

    nearby_sample_kdtree = spatial.cKDTree(sample_point_cloud.locations[nearby_sample_idxs])
    delta_length = 10.0 # in meters
    x,y = simplified_track.xy
    new_x = []
    new_y = []
    for i in range(0, len(x)-1):
        new_x.append(x[i])
        new_y.append(y[i])
        vec = np.array([x[i+1],y[i+1]]) - np.array([x[i], y[i]])
        length = np.linalg.norm(vec)
        if length < 1.0:
            continue
        n_pt_to_insert = int(length / delta_length + 0.5)
        if n_pt_to_insert == 0:
            continue
        delta_length = length / n_pt_to_insert
        vec /= length
        for j in range(0, n_pt_to_insert):
            pt = np.array([x[i], y[i]]) + vec*delta_length*(j+1)
            new_x.append(pt[0])
            new_y.append(pt[1])
    new_x.append(x[-1])
    new_y.append(y[-1])

    smoothed_pts = []
    extracted_segments = []
    for i in np.arange(len(new_x)):
        pt = np.array([new_x[i], new_y[i]])
        tmp_nearby_idxs = nearby_sample_kdtree.query_ball_point(pt, 25.0)
        if len(tmp_nearby_idxs) == 0:
            if len(smoothed_pts) > 1:
                linestring = LineString(smoothed_pts)
                simplified_linestring = linestring.simplify(10.0)
                extracted_segments.append(simplified_linestring)
                smoothed_pts = []
            continue
        else:
            nb_sample_idxs = nearby_sample_idxs[tmp_nearby_idxs]
            avg_pt = np.average(sample_point_cloud.locations[nb_sample_idxs], axis=0)
            smoothed_pts.append((avg_pt[0], avg_pt[1]))

    if len(smoothed_pts) > 1:
        linestring = LineString(smoothed_pts)
        simplified_linestring = linestring.simplify(10.0)
        extracted_segments.append(simplified_linestring)
  
    road_width = 0.0
    for seg_idx in nearby_seg_idxs:
        road_width += road_segments[seg_idx].half_width
    road_width /= len(nearby_seg_idxs)

    road_patches = []
    for linestring in extracted_segments:
        new_road_patch = RoadPatch()
        px,py = linestring.xy
        new_road_patch.center_line = np.array([px, py]).T
        segment_widths = [road_width]*(len(px)-1)
        new_road_patch.road_widths = np.array(segment_widths)
        road_patches.append(new_road_patch)
    
    return road_patches

def track_induced_segment_graph(tracks, 
                                road_segments):
    """ Generate a graph by projecting tracks on to road segments.
        Args:
            - tracks
            - road_segments
        Return:
            - G: as a directed networkx graph
                - Nodes: integer as index for each road segment in road_segments
                - Edges: a value where the connection may occur relative to the reference edge
    """
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(len(road_segments)))
    count = 0
    for track in tracks:
        print "now at ", count
        count += 1
        nearby_seg_idxs = track_to_segment_projction(track, road_segments)
        for idx in range(1, len(nearby_seg_idxs)-1):
            if G.has_edge(nearby_seg_idxs[idx], nearby_seg_idxs[idx+1]):
                G[nearby_seg_idxs[idx]][nearby_seg_idxs[idx+1]]['count'] += 1.0
            else:
                G.add_edge(nearby_seg_idxs[idx], nearby_seg_idxs[idx+1], count=1.0)

    for edge in G.edges():
        weight = 1.0 / G[edge[0]][edge[1]]['count']
        G[edge[0]][edge[1]]['weight'] = weight

    return G

def generate_segment_graph(road_segments,
                           search_radius = 50.0):
    """ Generate road segment graph
        Args:
            - road_segments: a list of RoadSegment objects
            - search_radius: searching for nearby road segments
        Return:
            - G: as a networkx graph
                - Nodes: integer as index for each road segment in road_segments
                - Edges: a value where the connection may occur relative to the reference edge
    """
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(len(road_segments)))
    # Creating a list of road
    road_segment_linestrings = []
    ANGLE_THRESHOLD = np.pi/12.0
    for r_seg_idx in np.arange(len(road_segments)):
        r_seg = road_segments[r_seg_idx]
        r_start = r_seg.center - r_seg.half_length*r_seg.direction
        r_end = r_seg.center + r_seg.half_length*r_seg.direction
        r_linestring = LineString([r_start, r_end])
        road_segment_linestrings.append(r_linestring)

    for cur_seg_idx in np.arange(len(road_segments)):
        p = road_segments[cur_seg_idx].center - road_segments[cur_seg_idx].half_length*road_segments[cur_seg_idx].direction
        q = road_segments[cur_seg_idx].center + road_segments[cur_seg_idx].half_length*road_segments[cur_seg_idx].direction
        a = p[0]*q[1] - p[1]*q[0]
        b = p[1] - q[1]
        c = q[0] - p[0]
        
        for nxt_seg_idx in np.arange(len(road_segments)):
            connect_loc = 0.0
            if nxt_seg_idx == cur_seg_idx:
                continue
            if road_segment_linestrings[cur_seg_idx].distance(road_segment_linestrings[nxt_seg_idx]) < search_radius:
                # Compute angle similarity
                angle_similarity = np.dot(road_segments[cur_seg_idx].direction,
                                          road_segments[nxt_seg_idx].direction)
                if angle_similarity < -1*np.cos(ANGLE_THRESHOLD):
                    continue
                elif angle_similarity > np.cos(ANGLE_THRESHOLD):
                    connect_loc = -1.0 # Value for merging
                else:
                    # Trying to make connections by adding edges
                    p1 = road_segments[nxt_seg_idx].center - road_segments[nxt_seg_idx].half_length*road_segments[nxt_seg_idx].direction
                    q1 = road_segments[nxt_seg_idx].center + road_segments[nxt_seg_idx].half_length*road_segments[nxt_seg_idx].direction
                    d = p1[0]*q1[1] - p1[1]*q1[0]
                    e = p1[1] - q1[1]
                    f = q1[0] - p1[0]
                    
                    alpha = b*f - c*e
                    px = d*c - a*f
                    py = a*e - b*d
                    if alpha < 0.1:
                        alpha = 0.0
                        px = 1.0
                        py = 1.0
                    else:
                        px /= alpha
                        py /= alpha
                        alpha = 1.0

                    if alpha > 0.0:
                        """
                            This is a valid intersection location, compute its location relatively 
                            to the current road segment frame
                        """
                        vec = np.array([px, py]) - p
                        len_proj = np.dot(vec, road_segments[cur_seg_idx].direction)
                        if len_proj < 0:
                            continue
                        connect_loc = len_proj

                G.add_weighted_edges_from([(cur_seg_idx, nxt_seg_idx, connect_loc)])
    return G

def main():
    parser = OptionParser()
    parser.add_option("-s", "--sample_point_cloud", dest="sample_point_cloud", help="Input sample point cloud filename", metavar="SAMPLE_POINT_CLOUD", type="string")
    parser.add_option("-r", "--road_segment", dest="road_segment", help="Input road segment filename", metavar="ROAD_SEGMENT", type="string")
    parser.add_option("-t", "--track", dest="tracks", help="Input GPS track file", metavar="TRACK_FILE", type="string")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)
    (options, args) = parser.parse_args()
    
    if not options.sample_point_cloud:
        parser.error("Input sample_point_cloud filename not found!")
    if not options.road_segment:
        parser.error("Input road segment file not found!")
    if not options.tracks:
        parser.error("Input GPS Track file not specified.")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    with open(options.sample_point_cloud, 'rb') as fin:
        sample_point_cloud = cPickle.load(fin)

    with open(options.road_segment, 'rb') as fin:
        road_segments = cPickle.load(fin)

    tracks = gps_track.load_tracks(options.tracks)

    # Compute points on road segments
    sample_idx_on_roads = {}
    sample_pt_kdtree = spatial.cKDTree(sample_point_cloud.locations)
    SEARCH_RADIUS = 30.0
    ANGLE_THRESHOLD = np.pi / 6.0
    for seg_idx in range(0, len(road_segments)):
        segment = road_segments[seg_idx]
        sample_idx_on_roads[seg_idx] = set([])
        start_pt = segment.center - segment.half_length*segment.direction
        end_pt = segment.center + segment.half_length*segment.direction
        n_pt_to_add = int(1.5 * segment.half_length / SEARCH_RADIUS + 0.5)
        px = np.linspace(start_pt[0], end_pt[0], n_pt_to_add)
        py = np.linspace(start_pt[1], end_pt[1], n_pt_to_add)
        nearby_sample_idxs = []
        for i in range(0, n_pt_to_add):
            pt = np.array([px[i], py[i]])
            tmp_idxs = sample_pt_kdtree.query_ball_point(pt, SEARCH_RADIUS)
            nearby_sample_idxs.extend(tmp_idxs)
        nearby_sample_idxs = set(nearby_sample_idxs)
        for sample_idx in nearby_sample_idxs:
            if np.dot(sample_point_cloud.directions[sample_idx], segment.direction) < np.cos(ANGLE_THRESHOLD):
                continue
            vec = sample_point_cloud.locations[sample_idx] - segment.center
            if abs(np.dot(vec, segment.norm_dir)) <= segment.half_width:
                sample_idx_on_roads[seg_idx].add(sample_idx)

    segment_graph_to_map(tracks,
                         road_segments,
                         sample_point_cloud)
                         
    return


    all_road_patches = []
    for selected_track_idx in range(0,100):
        print selected_track_idx
        road_patches = generate_road_patch_from_track(tracks[selected_track_idx],
                                                      road_segments,
                                                      sample_point_cloud,
                                                      sample_idx_on_roads)
        all_road_patches.extend(road_patches)

    arrow_params = const.arrow_params
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    #ax.plot(sample_point_cloud.locations[:,0], 
    #        sample_point_cloud.locations[:,1], 
    #        '.', color='gray')
    count = 0
    for road_patch in all_road_patches:
        color = const.colors[count%7]
        count += 1
        polygon = road_patch.road_polygon()
        for i in np.arange(len(road_patch.center_line)-1):
            if np.linalg.norm(road_patch.directions[i]) < 0.1:
                continue

            ax.arrow(road_patch.center_line[i,0],
                     road_patch.center_line[i,1],
                     10*road_patch.directions[i,0],
                     10*road_patch.directions[i,1],
                     width=0.5, head_width=4, fc=color, ec=color,
                     head_length=6, overhang=0.5, **arrow_params)

        patch = PolygonPatch(polygon, facecolor=color, edgecolor=color, alpha=0.5, zorder=4)
        ax.add_patch(patch)
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show()
    return

    track_on_road = project_tracks_to_road(tracks, road_segments)
    compute_segment_graph = False
    if compute_segment_graph:
        segment_graph = track_induced_segment_graph(tracks, road_segments)
        nx.write_gpickle(segment_graph, "test_segment_graph.gpickle")
    else:
        segment_graph = nx.read_gpickle("test_segment_graph.gpickle")
   
    max_node_count = -np.inf
    max_node = -1
    for node in segment_graph.nodes():
        out_edges = segment_graph.out_edges(node)
        sum_val = 0.0
        for edge in out_edges:
            sum_val += segment_graph[edge[0]][edge[1]]['count']

        if sum_val > max_node_count:
            max_node_count = sum_val
            max_node = node

    print "Totally %d edges."%(len(segment_graph.edges()))
    #segment_graph = generate_segment_graph(road_segments)
    arrow_params = const.arrow_params
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(sample_point_cloud.locations[:,0], 
            sample_point_cloud.locations[:,1], 
            '.', color='gray')
    #ax.plot([pt[0] for pt in selected_track.utm],
    #        [pt[1] for pt in selected_track.utm], 'r.-')
    
    #for track_idx in track_on_road[selected_seg_idx]:
    #    ax.plot([pt[0] for pt in tracks[track_idx].utm],
    #            [pt[1] for pt in tracks[track_idx].utm], '.')
    segment = road_segments[max_node]
    p0 = segment.center - segment.half_length*segment.direction + segment.half_width*segment.norm_dir
    p1 = segment.center + segment.half_length*segment.direction + segment.half_width*segment.norm_dir
    p2 = segment.center + segment.half_length*segment.direction - segment.half_width*segment.norm_dir
    p3 = segment.center - segment.half_length*segment.direction - segment.half_width*segment.norm_dir
    ax.plot([p0[0], p1[0]], [p0[1],p1[1]], 'r-')
    ax.plot([p1[0], p2[0]], [p1[1],p2[1]], 'r-')
    ax.plot([p2[0], p3[0]], [p2[1],p3[1]], 'r-')
    ax.plot([p3[0], p0[0]], [p3[1],p0[1]], 'r-')
    arrow_p0 = segment.center - segment.half_length*segment.direction
    ax.arrow(arrow_p0[0],
             arrow_p0[1],
             2*segment.half_length*segment.direction[0],
             2*segment.half_length*segment.direction[1],
             width=4, head_width=20, fc='r', ec='r',
             head_length=40, overhang=0.5, **arrow_params)

    for seg_idx in segment_graph.successors(max_node):
        segment = road_segments[seg_idx]
        p0 = segment.center - segment.half_length*segment.direction + segment.half_width*segment.norm_dir
        p1 = segment.center + segment.half_length*segment.direction + segment.half_width*segment.norm_dir
        p2 = segment.center + segment.half_length*segment.direction - segment.half_width*segment.norm_dir
        p3 = segment.center - segment.half_length*segment.direction - segment.half_width*segment.norm_dir
        ax.plot([p0[0], p1[0]], [p0[1],p1[1]], 'b-')
        ax.plot([p1[0], p2[0]], [p1[1],p2[1]], 'b-')
        ax.plot([p2[0], p3[0]], [p2[1],p3[1]], 'b-')
        ax.plot([p3[0], p0[0]], [p3[1],p0[1]], 'b-')
        arrow_p0 = segment.center - segment.half_length*segment.direction
        ax.arrow(arrow_p0[0],
                 arrow_p0[1],
                 2*segment.half_length*segment.direction[0],
                 2*segment.half_length*segment.direction[1],
                 width=4, head_width=20, fc='b', ec='b',
                 head_length=40, overhang=0.5, **arrow_params)

    #for i in np.arange(len(nearby_seg_idxs)):
    #    seg_idx = nearby_seg_idxs[i]
    #    segment = road_segments[seg_idx]
    #    arrow_p0 = segment.center - segment.half_length*segment.direction
    #    color = const.colors[i%7]
    #    #if segment_graph[selected_seg_idx][seg_idx]['weight'] == -1.0:
    #    #    color = 'g'
    #    ax.arrow(arrow_p0[0],
    #             arrow_p0[1],
    #             2*segment.half_length*segment.direction[0],
    #             2*segment.half_length*segment.direction[1],
    #             width=2, head_width=10, fc=color, ec=color,
    #             head_length=20, overhang=0.5, **arrow_params)

    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show()

    return

if __name__ == "__main__":
    sys.exit(main())
