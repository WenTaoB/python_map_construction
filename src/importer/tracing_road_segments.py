#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import sys
import cPickle
import math
import random
import time
import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.collections import LineCollection
from scipy import spatial
from scipy import signal

from skimage.transform import hough_line,hough_line_peaks, probabilistic_hough_line
from skimage.filter import canny
from skimage.morphology import skeletonize

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import gps_track
from point_cloud import PointCloud
import l1_skeleton_extraction

import const

def filter_point_cloud_using_grid(point_cloud, 
                                  point_directions,
                                  sample_grid_size,
                                  loc,
                                  R):
    """ Sample the input point cloud using a uniform grid. If there are points in the cell,
        we will use the average.
    """
    min_easting = loc[0]-R
    max_easting = loc[0]+R
    min_northing = loc[1]-R
    max_northing = loc[1]+R

    n_grid_x = int((max_easting - min_easting)/sample_grid_size + 0.5)
    n_grid_y = int((max_northing - min_northing)/sample_grid_size + 0.5)
    
    if n_grid_x > 1E4 or n_grid_y > 1E4:
        print "ERROR! The sampling grid is too small!"
        sys.exit(1)
    
    sample_points = []
    #sample_directions = []
    sample_canonical_directions = []

    geo_hash = {}
    dir_hash = {}
    geo_hash_count = {} 
    geo_hash_direction = {}
    for pt_idx in range(0, len(point_cloud.locations)):
        pt = np.copy(point_cloud.locations[pt_idx])
        #pt_dir = point_cloud.directions[pt_idx]

        px = int((pt[0] - min_easting) / sample_grid_size)
        py = int((pt[1] - min_northing) / sample_grid_size)

        if px<0 or px>n_grid_x or py<0 or py>n_grid_y:
            print "ERROR! Point outside the grid!"
            sys.exit(1)

        if geo_hash.has_key((px, py)):
            geo_hash_count[(px, py)] += 1
            geo_hash[(px, py)] += pt
            #dir_hash[(px, py)] += pt_dir
            for direction in point_directions[pt_idx]:
                geo_hash_direction[(px, py)].append(np.copy(direction))
        else:
            geo_hash_count[(px, py)] = 1.0
            geo_hash_direction[(px, py)] = []
            for direction in point_directions[pt_idx]:
                geo_hash_direction[(px, py)].append(np.copy(direction))

            geo_hash[(px, py)] = pt
            #dir_hash[(px, py)] = pt_dir
    
    for key in geo_hash.keys():
        pt = geo_hash[key] / geo_hash_count[key]
        #dir_hash[key] /= geo_hash_count[key]

        sample_points.append(pt)
        #sample_directions.append(dir_hash[key])

        directions = []
        for direction in geo_hash_direction[key]:
            if len(directions) == 0:
                directions.append(direction)
            else:
                found_match = False
                for idx in range(0, len(directions)):
                    dot_value = np.dot(direction, directions[idx])
                    if abs(dot_value) > 0.7:
                        found_match = True
                        break
                if not found_match:
                    directions.append(np.copy(direction))
        sample_canonical_directions.append(list(directions))

    sample_point_cloud = PointCloud(sample_points, [-1]*len(sample_points), [-1]*len(sample_points))
    return sample_point_cloud, sample_canonical_directions

def grow_segment(starting_pt_idx,
                 sample_point_cloud, 
                 sample_point_kdtree,
                 sample_point_directions,
                 search_radius,
                 width_threshold,
                 angle_threshold,
                 min_pt_to_record):
    """ Grow segment from a starting point.
        Args:
            - starting_pt_idx: index in sample_point_cloud.locations
            - sample_point_kdtree: kdtree built from sample_point_cloud.locations
            - sample_point_directions: principal_directions for each sample points
            - search_radius: search radius each time to grow the segment
            - width_threshold: distance threshold to the line defined by the searching point
                               and its principal direction
            - angle_threshold: in radius, largest angle tolerance 
        Return:
            - result_segment_pt_idxs: a list of indices recording the sample points in the resulting segment
    """
    result_segments = []
    # Search through all its directions
    for direction in sample_point_directions[starting_pt_idx]:
        result_segment_pt_idxs = []
        segment_point_idxs_dict = {}
        segment_point_potential = {} # a rough sorting of the extracted points

        if np.linalg.norm(direction) < 0.1:
            continue
        front_pt_idx = starting_pt_idx
        end_pt_idx = starting_pt_idx
        front_stopped = False
        end_stopped = False
        segment_point_idxs_dict[starting_pt_idx] = 0.0
        front_dir = np.copy(direction)
        end_dir = np.copy(direction)
        front_potential = 0.0
        end_potential = 0.0
        while True:
            if not front_stopped:
                front_potential += 1.0
                candidate_nearby_point_idxs = \
                    sample_point_kdtree.query_ball_point(sample_point_cloud.locations[front_pt_idx], search_radius)
                norm_dir = np.array([-1*direction[1], direction[0]])
                nxt_front_pt_idx = -1
                nxt_front_pt_proj = 0.0
                nxt_front_dir = np.array([[0.0, 0.0]])
                n_pt_to_add = 0
                for candidate_idx in candidate_nearby_point_idxs:
                    if candidate_idx == front_pt_idx:
                        continue
                    for pt_dir in sample_point_directions[candidate_idx]:
                        if np.dot(pt_dir, front_dir) < np.cos(angle_threshold):
                            continue
                        if abs(np.dot(front_dir, pt_dir)) > np.cos(angle_threshold):
                            vec = sample_point_cloud.locations[candidate_idx] - sample_point_cloud.locations[front_pt_idx]
                            pt_proj = np.dot(vec, front_dir)
                            if pt_proj > 0 and abs(np.dot(norm_dir, vec)) <= width_threshold:
                                if not segment_point_idxs_dict.has_key(candidate_idx):
                                    segment_point_idxs_dict[candidate_idx] = front_potential
                                    vec_norm = np.linalg.norm(vec)
                                    if vec_norm <= 0.001:
                                        segment_point_idxs_dict[candidate_idx] += pt_proj
                                    else:
                                        segment_point_idxs_dict[candidate_idx] += pt_proj/vec_norm
                                    if pt_proj > nxt_front_pt_proj:
                                        nxt_front_pt_proj = pt_proj
                                        nxt_front_pt_idx = candidate_idx
                                        nxt_front_dir = np.copy(pt_dir)
                                        n_pt_to_add += 1
                front_pt_idx = nxt_front_pt_idx
                front_dir = nxt_front_dir
                
                if n_pt_to_add == 0:
                    front_stopped = True

            if not end_stopped:
                end_potential -= 1.0
                candidate_nearby_point_idxs = \
                        sample_point_kdtree.query_ball_point(sample_point_cloud.locations[front_pt_idx], search_radius)
                norm_dir = np.array([-1*direction[1], direction[0]])
                nxt_end_pt_idx = -1
                nxt_end_pt_proj = 0.0
                nxt_end_dir = np.array([[0.0, 0.0]])
                n_pt_to_add = 0
                for candidate_idx in candidate_nearby_point_idxs:
                    if candidate_idx == end_pt_idx:
                        continue
                    for pt_dir in sample_point_directions[candidate_idx]:
                        if np.dot(pt_dir, end_dir) < np.cos(angle_threshold):
                            continue
                        if abs(np.dot(end_dir, pt_dir)) > np.cos(angle_threshold):
                            vec = sample_point_cloud.locations[candidate_idx] - sample_point_cloud.locations[end_pt_idx]
                            pt_proj = np.dot(vec, -1*end_dir)
                            if pt_proj > 0 and abs(np.dot(norm_dir, vec)) <= width_threshold:
                                if not segment_point_idxs_dict.has_key(candidate_idx):
                                    segment_point_idxs_dict[candidate_idx] = end_potential
                                    vec_norm = np.linalg.norm(vec)
                                    if vec_norm <= 0.001:
                                        segment_point_idxs_dict[candidate_idx] -= pt_proj
                                    else:
                                        segment_point_idxs_dict[candidate_idx] -= pt_proj/vec_norm

                                    if pt_proj > nxt_end_pt_proj:
                                        nxt_end_pt_proj = pt_proj
                                        nxt_end_pt_idx = candidate_idx
                                        nxt_end_dir = np.copy(pt_dir)
                                        n_pt_to_add += 1

                end_pt_idx = nxt_end_pt_idx
                end_dir = nxt_end_dir
                if n_pt_to_add == 0:
                    end_stopped = True
            if front_stopped and end_stopped:
                break
        
        if len(segment_point_idxs_dict.keys()) >= min_pt_to_record:
            tmp_seg_idxs = []
            tmp_seg_order = []
            for key in segment_point_idxs_dict.keys():
                tmp_seg_idxs.append(key)
                tmp_seg_order.append(segment_point_idxs_dict[key])
            tmp_seg_order = np.array(tmp_seg_order)
            sorted_idx = np.argsort(tmp_seg_order)
            result_segment_pt_idxs = []
            for i in range(0, len(sorted_idx)):
                result_segment_pt_idxs.append(tmp_seg_idxs[sorted_idx[i]])
            result_segments.append(result_segment_pt_idxs)

    return result_segments

def main():
       
    compute_canonical_dir = False
    
    GRID_SIZE = 2.5 # in meters

    # Target location and radius
    # test_point_cloud.dat
    #LOC = (447772, 4424300)
    #R = 500

    # test_point_cloud1.dat
    LOC = (446458, 4422150)
    R = 500

    compute_sample_point_cloud = False
    if compute_sample_point_cloud:
        if len(sys.argv) != 5:
            print "ERROR! Correct usage is:"
            return

        with open(sys.argv[1], "rb") as fin:
            point_cloud = cPickle.load(fin)
        print "there are %d points in the point cloud."%point_cloud.locations.shape[0]
        #tracks = gps_track.load_tracks(sys.argv[2])

        with open(sys.argv[2], 'rb') as fin:
            point_directions = cPickle.load(fin)
       
        # Grid sample
        sample_point_cloud, sample_canonical_directions =\
                             filter_point_cloud_using_grid(point_cloud, 
                                                           point_directions,
                                                           10,
                                                           LOC,
                                                           R)

        # Correct direction using tracks 
        # build sample point kdtree
        sample_point_kdtree = spatial.cKDTree(sample_point_cloud.locations)
        expanded_directions = []
        votes_directions = []
        for i in range(0, len(sample_canonical_directions)):
            directions = []
            votes = []
            for direction in sample_canonical_directions[i]:
                directions.append(direction)
                votes.append(0)
                directions.append(-1*direction)
                votes.append(0)
            expanded_directions.append(directions)
            votes_directions.append(votes)

        for i in range(0, point_cloud.locations.shape[0]):
            # find nearby sample point
            dist, sample_idx = sample_point_kdtree.query(point_cloud.locations[i])
            for direction_idx in range(0, len(expanded_directions[sample_idx])):
                direction = expanded_directions[sample_idx][direction_idx]
                dot_product = np.dot(direction, point_cloud.directions[i])
                if dot_product >= 0.866:
                    votes_directions[sample_idx][direction_idx] += 1

        threshold = 1
        revised_canonical_directions = []
        for i in range(0, len(expanded_directions)):
            revised_dir = []
            for dir_idx in range(0, len(expanded_directions[i])):
                if votes_directions[i][dir_idx] >= threshold:
                    revised_dir.append(expanded_directions[i][dir_idx])
            revised_canonical_directions.append(revised_dir) 

        with open(sys.argv[3], 'wb') as fout:
            cPickle.dump(sample_point_cloud, fout, protocol=2)

        with open(sys.argv[4], 'wb') as fout:
            cPickle.dump(revised_canonical_directions, fout, protocol=2)
        return

    else:
        if len(sys.argv) != 3:
            print "ERROR! Correct usage is:"
            return
        with open(sys.argv[1], 'rb') as fin:
            sample_point_cloud = cPickle.load(fin)

        with open(sys.argv[2], 'rb') as fin:
            revised_canonical_directions = cPickle.load(fin)
    #visualize_sample_point_cloud(sample_point_cloud, 
    #                             revised_canonical_directions,
    #                             point_cloud, LOC, R)
    SEARCH_RADIUS = 25
    WIDTH_THRESHOLD = 15
    ANGLE_THRESHOLD = np.pi / 4.0
    MIN_PT_TO_RECORD = 5
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    sample_point_kdtree = spatial.cKDTree(sample_point_cloud.locations)
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}

    #for i in range(0, sample_point_cloud.locations.shape[0]):
    #    for direction in revised_canonical_directions[i]:
    #        ax.arrow(sample_point_cloud.locations[i][0],
    #                 sample_point_cloud.locations[i][1],
    #                 20*direction[0], 
    #                 20*direction[1],
    #                 width=0.5, head_width=5, fc='gray', ec='gray',
    #                 head_length=10, overhang=0.5, **arrow_params)

    point_count = np.zeros(sample_point_cloud.locations.shape[0])
    count = 0
    print "there are %d sample points."%sample_point_cloud.locations.shape[0]
    start_time = time.time()
    for starting_pt_idx in range(0, sample_point_cloud.locations.shape[0]):
    #for starting_pt_idx in range(3, 5): 
        if point_count[starting_pt_idx] > len(revised_canonical_directions[starting_pt_idx]):
            continue

        result_segments = grow_segment(starting_pt_idx,
                                       sample_point_cloud,
                                       sample_point_kdtree,
                                       revised_canonical_directions,
                                       SEARCH_RADIUS,
                                       WIDTH_THRESHOLD,
                                       ANGLE_THRESHOLD,
                                       MIN_PT_TO_RECORD)
        
        #ax.plot(sample_point_cloud.locations[starting_pt_idx,0],
        #        sample_point_cloud.locations[starting_pt_idx,1], 'or')

        for i in range(0, len(result_segments)):
            count += 1
            segment_point_idxs = result_segments[i]
            point_count[segment_point_idxs] += 1
            color = const.colors[count%7]
            #ax.plot(sample_point_cloud.locations[segment_point_idxs[-1],0],
            #        sample_point_cloud.locations[segment_point_idxs[-1],1],
            #        'o', color=const.colors[count%7], markersize=12)
            points = np.copy(sample_point_cloud.locations[segment_point_idxs])
            fitted_curve = l1_skeleton_extraction.skeleton_extraction(points)
            #sys.exit(1)
            ax.plot(fitted_curve[:,0], fitted_curve[:, 1], '-', color=color)
            ax.arrow(fitted_curve[-2,0], fitted_curve[-2,1],
                     fitted_curve[-1,0]-fitted_curve[-2,0],
                     fitted_curve[-1,1]-fitted_curve[-2,1],
                     width=0.5, head_width=5, fc=color, ec=color,
                     head_length=10, overhang=0.5, **arrow_params)

    end_time = time.time()
    print "Time elapsed: %d"%(int(end_time-start_time))
    print "There are %d segments."%count
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show()    
    return

if __name__ == "__main__":
    sys.exit(main())
