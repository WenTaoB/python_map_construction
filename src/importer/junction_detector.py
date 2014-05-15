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

class SamplePatch:
    def __init__(self, 
                 sample_idxs, 
                 directions,
                 sample_point_cloud):
        """
            Args:
                - sample_idxs: a list of indices of the sample points
                - direction: a list of ndarray for the corresponding direction of the sample
        """
        self.sample_idxs = {}
        canonical_dir = np.array([0.0, 0.0])
        for idx in range(0, len(sample_idxs)):
            sample_idx = sample_idxs[idx]
            self.sample_idxs[sample_idx] = directions[idx]
            canonical_dir += directions[idx]
        canonical_dir_norm = np.linalg.norm(canonical_dir)  
        canonical_dir /= canonical_dir_norm
        self.direction = canonical_dir
        self.center = np.mean(sample_point_cloud.locations[sample_idxs], axis=0)

    def min_distance(self, pt, sample_point_cloud):
        points = np.array(sample_point_cloud.locations[self.sample_idxs.keys()])
        vecs = points - pt
        dist = np.inf
        min_pt_idx = -1
        for pt_idx in self.sample_idxs.keys():
            vec = sample_point_cloud.locations[pt_idx] - pt
            vec_norm = np.linalg.norm(vec)
            if vec_norm < dist:
                dist = vec_norm
                min_pt_idx = pt_idx
        return dist, min_pt_idx

    def insert_pt(self, pt_idx, sample_point_cloud):
        new_center = len(self.sample_idxs.keys())*self.center + sample_point_cloud.locations[pt_idx]
        self.sample_idxs[pt_idx] = 1
        self.center = new_center / len(self.sample_idxs.keys())

    def start_end(self, sample_point_cloud):
        points = np.copy(sample_point_cloud.locations[self.sample_idxs.keys()]) - self.center
        if np.linalg.norm(self.direction) == 0:
            return []
        projections = np.dot(points, self.direction)
        min_proj = min(projections)
        max_proj = max(projections)
        start = self.center + min_proj*self.direction
        end = self.center + max_proj*self.direction
        return [start, end]

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

def main():
    compute_canonical_dir = False
    
    GRID_SIZE = 2.5 # in meters

    # Target location and radius
    # test_point_cloud.dat
    LOC = (447772, 4424300)
    R = 500

    # test_point_cloud1.dat
    #LOC = (446458, 4422150)
    #R = 500

    # San Francisco
    #LOC = (551281, 4180430) 
    #R = 500

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
    
    SEARCH_RADIUS = 100
    WIDTH_THRESHOLD = 25
    ANGLE_THRESHOLD = np.pi / 6.0
    MIN_PT_TO_RECORD = 5
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    sample_point_kdtree = spatial.cKDTree(sample_point_cloud.locations)
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}

    sample_patches = []
    killed_sample = {}
    sample_flags = []
    for sample_idx in range(0, sample_point_cloud.locations.shape[0]):
        flag = []
        for dir_idx in range(0, len(revised_canonical_directions[sample_idx])):
            flag.append(0)
        sample_flags.append(flag)

    while True:
        for sample_idx in range(0, sample_point_cloud.locations.shape[0]):
            if killed_sample.has_key(sample_idx):
                continue

            sample_pt = sample_point_cloud.locations[sample_idx]

            # Search a neighborhood 
            neighbor_idxs = sample_point_kdtree.query_ball_point(sample_pt, SEARCH_RADIUS)
            if len(neighbor_idxs) == 0:
                filtered_sample_idxs = [sample_idx]
                if len(revised_canonical_directions[sample_idx]) == 0:
                    filtered_sample_dirs = [np.array([0.0, 0.0])]
                else:
                    filtered_sample_dirs = [revised_canonical_directions[sample_idx][0]]

                new_sample_patch = SamplePatch(filtered_sample_idxs,
                                               filtered_sample_dirs,
                                               sample_point_cloud)
                sample_patches.append(new_sample_patch)
                killed_sample[sample_idx] = 1

            # Check if it can be inserted into existing sample_patches
            if len(revised_canonical_directions[sample_idx]) == 0:
                for existing_patch in sample_patches:
                    dist, min_pt_idx = existing_patch.min_distance(sample_pt, sample_point_cloud)
                    if dist < SEARCH_RADIUS:
                        vec = sample_pt - sample_point_cloud.locations[min_pt_idx]
                        if abs(np.dot(vec, sample_norm)) > WIDTH_THRESHOLD:
                            continue
                        
                        if len(revised_canonical_directions[sample_idx]) == 0:
                            killed_sample[sample_idx] = 1
                            existing_patch.insert_pt(sample_idx, sample_point_cloud)
                            break

                        for sample_dir_idx in range(0, len(revised_canonical_directions[sample_idx])):
                            sample_dir = revised_canonical_directions[sample_idx][sample_dir_idx]
                            if np.dot(sample_dir, existing_patch.direction) > ANGLE_THRESHOLD:
                                existing_patch.insert_pt(sample_idx, sample_point_cloud)
                                sample_flags[sample_idx][sample_dir_idx] = 1
                                break
            
            for flag in sample_flags[sample_idx]:
                if flag == 0:
                    has_unmarked_flag = True
            if not has_unmarked_flag:
                killed_sample[sample_idx] = 1
                continue

            # Iterate over all its directions
            for sample_dir_idx in range(0, len(revised_canonical_directions[sample_idx])):
                if sample_flags[sample_idx][sample_dir_idx] == 1:
                    continue
                sample_dir = revised_canonical_directions[sample_idx][sample_dir_idx]
                sample_norm = np.array([-1*sample_dir[1], sample_dir[0]])
                filtered_sample_idxs = [sample_idx]
                filtered_sample_dirs = [sample_dir]
                sample_flags[sample_idx][sample_dir_idx] = 1
                # Check its neighbors, accept those whose angles are compatible
                for nb_idx in neighbor_idxs:
                    if killed_sample.has_key(nb_idx):
                        continue
                    # Traverse all its directions
                    n_dir = len(revised_canonical_directions[nb_idx])
                    vec = sample_point_cloud.locations[nb_idx] - sample_pt
                    width_dist = abs(np.dot(vec, sample_norm))
                    if width_dist > WIDTH_THRESHOLD:
                        continue

                    if n_dir == 0:
                        # This neighbor point is compatible
                        filtered_sample_idxs.append(nb_idx)
                        filtered_sample_dirs.append(np.array([0.0, 0.0]))
                        killed_sample[nb_idx] = 1
                        break

                    for nb_dir_idx in range(0, len(revised_canonical_directions[nb_idx])):
                        if sample_flags[nb_idx][nb_dir_idx] == 1:
                            continue
                        nb_dir = revised_canonical_directions[nb_idx][nb_dir_idx]
                        if np.dot(nb_dir, sample_dir) > np.cos(ANGLE_THRESHOLD):
                            # Direction compatible
                            filtered_sample_idxs.append(nb_idx)
                            filtered_sample_dirs.append(nb_dir)
                            sample_flags[nb_idx][nb_dir_idx] = 1
                    has_unmarked_flag = False 
                    for flag in sample_flags[nb_idx]:
                        if flag == 0:
                            has_unmarked_flag = True
                    if not has_unmarked_flag:
                        killed_sample[nb_idx] = 1
            
            has_unmarked_flag = False 
            for flag in sample_flags[sample_idx]:
                if flag == 0:
                    has_unmarked_flag = True
            if not has_unmarked_flag:
                killed_sample[sample_idx] = 1

            new_sample_patch = SamplePatch(filtered_sample_idxs, 
                                           filtered_sample_dirs,
                                           sample_point_cloud)
            sample_patches.append(new_sample_patch)

        print "total: %d, killed: %d"%(sample_point_cloud.directions.shape[0], len(killed_sample.keys()))
        if len(killed_sample.keys()) == sample_point_cloud.directions.shape[0]:
            break

    print "There are %d patches"%len(sample_patches)

    for patch_idx in range(0, len(sample_patches)):
        patch = sample_patches[patch_idx]
        dot_value = np.dot(patch.direction, np.array((1.0, 0.0)))
        angle = np.degrees(np.arccos(dot_value))
        if patch.direction[1] < 0:
            angle = 360 - angle
        if angle >= 0 and angle < 45:
            color_count = 0
        elif angle >= 45 and angle < 135:
            color_count = 1 
        elif angle >= 135 and angle < 225:
            color_count = 2
        elif angle >= 225 and angle < 315:
            color_count = 3
        elif angle >= 315 and angle < 360:
            color_count = 0
        if np.linalg.norm(patch.direction) == 0:
            color_count = 5

        color = const.colors[color_count]
        ax.plot(sample_point_cloud.locations[patch.sample_idxs.keys(),0],
                sample_point_cloud.locations[patch.sample_idxs.keys(),1],
                '.', color=color)
        result = patch.start_end(sample_point_cloud)
        if len(result) == 0:
            continue
        
        if np.linalg.norm(result[1] - result[0]) > 1.0:
            ax.arrow(result[0][0],
                     result[0][1],
                     result[1][0]-result[0][0], 
                     result[1][1]-result[0][1],
                     width=0.5, head_width=5, fc=color, ec=color,
                     head_length=10, overhang=0.5, **arrow_params)
        else:
            ax.plot(patch.center[0], patch.center[1], 'o', color=color)
            ax.arrow(patch.center[0],
                     patch.center[1],
                     patch.direction[0],
                     patch.direction[1],
                     width=0.5, head_width=5, fc=color, ec=color,
                     head_length=10, overhang=0.5, **arrow_params)

    #for i in range(0, sample_point_cloud.locations.shape[0]):
    #    for direction in revised_canonical_directions[i]:
    #        ax.arrow(sample_point_cloud.locations[i][0],
    #                 sample_point_cloud.locations[i][1],
    #                 20*direction[0], 
    #                 20*direction[1],
    #                 width=0.5, head_width=5, fc='gray', ec='gray',
    #                 head_length=10, overhang=0.5, **arrow_params)

    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show()    
    return

if __name__ == "__main__":
    sys.exit(main())
