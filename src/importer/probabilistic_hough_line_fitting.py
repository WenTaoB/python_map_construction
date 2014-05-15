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

import const

class SampleCluster:
    def __init__(self, member_idxs, directions, sample_point_cloud):
        self.member_samples = {} # a list of indices of sample point
        direction = np.array([0.0, 0.0])
        mass_center = np.array([0.0, 0.0])
        for idx in range(0, len(member_idxs)):
            member_idx = member_idxs[idx]
            self.member_samples[member_idx] = 1
            direction += directions[idx]
            mass_center += sample_point_cloud.locations[member_idx]
        self.mass_center = mass_center / len(member_idxs)
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0.01:
            self.direction = direction / dir_norm
        else:
            self.direction = np.array([0.0, 0.0])

    def compute_similarity(self, other_sample_cluster):
        # Jaccard distance
        angle_similarity = np.dot(self.direction, other_sample_cluster.direction)
        if angle_similarity <= 0:
            angle_similarity = 0.0
        keys1 = self.member_samples.keys()
        keys2 = other_sample_cluster.member_samples.keys()
        intersection = np.intersect1d(keys1, keys2)
        union = np.union1d(keys1, keys2)
        if len(intersection) >= 0.9*len(keys1) or len(intersection) >= 0.9*len(keys2):
            similarity = angle_similarity
        else:
            similarity = angle_similarity * float(len(intersection)) / float(len(union))
        return similarity

    def merge_cluster(self, other_cluster):
        for key in other_cluster.member_samples.keys():
            self.member_samples[key] = 1
        direction = np.array([0.0, 0.0])
        direction += len(self.member_samples.keys())*self.direction 
        direction += len(other_cluster.member_samples.keys())*other_cluster.direction
        direction /= (len(self.member_samples.keys()) + len(other_cluster.member_samples.keys()))
        direction_norm = np.linalg.norm(direction)
    
        new_center = np.array([0.0, 0.0])
        new_center += len(self.member_samples.keys())*self.mass_center
        new_center += len(other_cluster.member_samples.keys())*other_cluster.mass_center
        self.mass_center = new_center / (len(self.member_samples.keys())+ len(other_cluster.member_samples.keys()))

        if direction_norm > 0.01:
            self.direction = direction / direction_norm
        else:
            self.direction = np.array([0.0, 0.0])

def grid_img(point_cloud,
             sample_grid_size,
             loc,
             R):
    """ Sample the input point cloud using a uniform grid. If there are points in the cell,
        we will use the average.
    """
    sample_points = []
    sample_directions = []
    
    min_easting = loc[0]-R
    max_easting = loc[0]+R
    min_northing = loc[1]-R
    max_northing = loc[1]+R

    n_grid_x = int((max_easting - min_easting)/sample_grid_size + 0.5)
    n_grid_y = int((max_northing - min_northing)/sample_grid_size + 0.5)

    results = np.zeros((n_grid_x+1, n_grid_y+1))

    if n_grid_x > 1E4 or n_grid_y > 1E4:
        print "ERROR! The sampling grid is too small!"
        sys.exit(1)
    
    geo_hash = {}
    dir_hash = {}
    geo_hash_count = {} 
    for pt_idx in range(0, len(point_cloud.locations)):
        pt = point_cloud.locations[pt_idx]
        pt_dir = point_cloud.directions[pt_idx]

        px = int((pt[0] - min_easting) / sample_grid_size)
        py = int((pt[1] - min_northing) / sample_grid_size)

        if px<0 or px>n_grid_x or py<0 or py>n_grid_y:
            print "ERROR! Point outside the grid!"
            sys.exit(1)
        if geo_hash.has_key((px, py)):
            geo_hash_count[(px, py)] += 1
            geo_hash[(px, py)] += pt
            dir_hash[(px, py)] += pt_dir
        else:
            geo_hash_count[(px, py)] = 1.0
            geo_hash[(px, py)] = np.copy(pt)
            dir_hash[(px, py)] = np.copy(pt_dir)
    
    for key in geo_hash.keys():
        results[key[0], key[1]] = 1
    
    return results

def visualize_sample_points(point_cloud, sample_point_cloud, loc, R):
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(121, aspect="equal")
    ax.plot(point_cloud.locations[:,0], point_cloud.locations[:,1], '.')
    ax.set_xlim([loc[0]-R, loc[0]+R]) 
    ax.set_ylim([loc[1]-R, loc[1]+R]) 

    ax = fig.add_subplot(122, aspect="equal")
    ax.plot(sample_point_cloud.locations[:,0], sample_point_cloud.locations[:,1], 'ro')
    ax.quiver(sample_point_cloud.locations[:,0],
              sample_point_cloud.locations[:,1],
              sample_point_cloud.directions[:,0],
              sample_point_cloud.directions[:,1]) 
    ax.set_xlim([loc[0]-R, loc[0]+R]) 
    ax.set_ylim([loc[1]-R, loc[1]+R]) 
    plt.show()
    
def remove_pixels(sample_img, 
                  lines, 
                  p_removal,
                  search_range):
    """ Remove pixels with p_removal
    """
    new_img = np.copy(sample_img)
    for line in lines:
        start_pixel = line[0]
        end_pixel = line[1]
        n_step = max(abs(start_pixel[0]-end_pixel[0]), abs(start_pixel[1]-end_pixel[1])) + 1
        x = np.linspace(start_pixel[0], end_pixel[0], n_step)
        y = np.linspace(start_pixel[1], end_pixel[1], n_step)
        nearby_pixels = {}
        for p in zip(x,y):
            pixel = (int(p[0]), int(p[1]))
            for px in range(pixel[0]-search_range, pixel[0]+search_range+1):
                if px < 0 or px > sample_img.shape[0]-1:
                    continue
                for py in range(pixel[1]-search_range, pixel[1]+search_range+1):
                    if py < 0 or py > sample_img.shape[1]-1:
                        continue
                    if new_img[py, px] == 1:
                        nearby_pixels[(py, px)] = 1
        
        for pixel in nearby_pixels.keys():
            prob = np.random.rand()
            if prob < p_removal:
                new_img[pixel[0], pixel[1]] = 0

    return new_img

def extract_line_segments(point_cloud, grid_size, loc, R, 
                          line_gap, search_range, p_removal):
    
    sample_img = grid_img(point_cloud, grid_size, loc, R)

    all_lines = []
    lines = probabilistic_hough_line(sample_img, 
                                     line_length=50,
                                     line_gap=15)
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111)
    ax.imshow(sample_img.T, cmap='gray')
    for line in lines:
        ax.plot([line[0][1], line[1][1]],
                [line[0][0], line[1][0]], 'r-', linewidth=2)

    ax.set_xlim([0, sample_img.shape[0]])
    ax.set_ylim([0, sample_img.shape[1]])
    plt.show()
    all_lines.extend(lines)
    modified_img1 = remove_pixels(sample_img, 
                                  lines, 
                                  p_removal=p_removal,
                                  search_range=search_range)

    new_lines1 = probabilistic_hough_line(modified_img1, 
                                         line_length=25,
                                         line_gap=10)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111)
    ax.imshow(modified_img1.T, cmap='gray')
    for line in new_lines1:
        ax.plot([line[0][1], line[1][1]],
                [line[0][0], line[1][0]], 'r-', linewidth=2)

    ax.set_xlim([0, sample_img.shape[0]])
    ax.set_ylim([0, sample_img.shape[1]])
    plt.show()
 
    all_lines.extend(new_lines1)
    modified_img2 = remove_pixels(modified_img1,
                                  new_lines1,
                                  p_removal=p_removal,
                                  search_range=search_range)

    new_lines2 = probabilistic_hough_line(modified_img2,
                                          line_length=10,
                                          line_gap=10)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111)
    ax.imshow(modified_img2.T, cmap='gray')
    for line in new_lines2:
        ax.plot([line[0][1], line[1][1]],
                [line[0][0], line[1][0]], 'r-', linewidth=2)

    ax.set_xlim([0, sample_img.shape[0]])
    ax.set_ylim([0, sample_img.shape[1]])
    plt.show()
 
    all_lines.extend(new_lines2)
    modified_img3 = remove_pixels(modified_img2,
                                  new_lines2,
                                  p_removal=p_removal,
                                  search_range=search_range)

    new_lines3 = probabilistic_hough_line(modified_img3,
                                          line_length=5,
                                          line_gap=5)
   
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111)
    ax.imshow(modified_img3.T, cmap='gray')
    for line in new_lines3:
        ax.plot([line[0][1], line[1][1]],
                [line[0][0], line[1][0]], 'r-', linewidth=2)

    ax.set_xlim([0, sample_img.shape[0]])
    ax.set_ylim([0, sample_img.shape[1]])
    plt.show()

    all_lines.extend(new_lines3)

    orig_lines = []
    for line in all_lines:
        line_start = line[0]
        line_end = line[1]
        start_e = line_start[1]*grid_size + loc[0] - R
        start_n = line_start[0]*grid_size + loc[1] - R

        end_e = line_end[1]*grid_size + loc[0] - R
        end_n = line_end[0]*grid_size + loc[1] - R
        
        orig_line1 = [(start_e, start_n), (end_e, end_n)]
        orig_lines.append(orig_line1)
        orig_line2 = [(end_e, end_n), (start_e, start_n)]
        orig_lines.append(orig_line2)

    return np.array(orig_lines)

def visualize_extracted_lines(point_cloud, lines, loc, R):
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(point_cloud.locations[:,0],
            point_cloud.locations[:,1],
            '.', color='gray')
    for line in lines:
        ax.plot([line[0][0], line[1][0]],
                [line[0][1], line[1][1]],
                '-')
    ax.set_xlim([loc[0]-R, loc[0]+R])
    ax.set_ylim([loc[1]-R, loc[1]+R])
    plt.show()

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

def visualize_sample_point_cloud(sample_point_cloud, 
                                 sample_canonical_directions,
                                 point_cloud, loc, R):
    fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(121, aspect='equal')
    #ax.plot(point_cloud.locations[:,0],
    #        point_cloud.locations[:,1],
    #        '.', color='gray')
    #ax.set_xlim([loc[0]-R, loc[0]+R])
    #ax.set_ylim([loc[1]-R, loc[1]+R])
    
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(sample_point_cloud.locations[:,0],
            sample_point_cloud.locations[:,1],
            '.', color='gray')
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    for i in range(0, sample_point_cloud.locations.shape[0]):
        for direction in sample_canonical_directions[i]:
            # direction angle
            dot_value = np.dot(direction, np.array((1.0, 0.0)))
            angle = np.degrees(np.arccos(dot_value))
            if direction[1] < 0:
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
            
            color = const.colors[color_count]
            ax.arrow(sample_point_cloud.locations[i][0],
                     sample_point_cloud.locations[i][1],
                     20*direction[0], 20*direction[1], fc=color, ec=color,
                     width=0.5, head_width=5,
                     head_length=10, overhang=0.5, **arrow_params)
    ax.set_xlim([loc[0]-R, loc[0]+R])
    ax.set_ylim([loc[1]-R, loc[1]+R])
 
    plt.show()

def main():
    if len(sys.argv) != 5:
        print "ERROR! Correct usage is:"
        return
    
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

    with open(sys.argv[1], "rb") as fin:
        point_cloud = cPickle.load(fin)
    print "there are %d points in the point cloud."%point_cloud.locations.shape[0]
    tracks = gps_track.load_tracks(sys.argv[2])

    compute_canonical_dir = True
    if compute_canonical_dir:
        LINE_GAP = 40
        SEARCH_RANGE = 5
        P_REMOVAL = 0.1
            
        lines = extract_line_segments(point_cloud,
                                      GRID_SIZE,
                                      LOC,
                                      R,
                                      LINE_GAP,
                                      SEARCH_RANGE,
                                      P_REMOVAL)

        visualize_extracted_lines(point_cloud, lines, LOC, R)
        return 

        line_vecs = []
        line_norms = []
        for line in lines:
            line_vec = np.array((line[1][0]-line[0][0], line[1][1]-line[0][1]))
            line_vec_norm = np.linalg.norm(line_vec)
            line_vec /= line_vec_norm
            line_vecs.append(line_vec)
            line_norm = np.array((-1*line_vec[1], line_vec[0]))
            line_norms.append(line_norm)
        line_vecs = np.array(line_vecs)
        line_norms = np.array(line_norms)

        #angle_distance = 1.1 - np.dot(line_vecs, line_vecs.T)
        dist_threshold = 5

        point_directions = []
        print "start computing"
        for pt_idx in range(0, point_cloud.locations.shape[0]):
            pt = point_cloud.locations[pt_idx]
            # search nearby lines
            vec1s = pt - lines[:,0]
            vec2s = pt - lines[:,1]
            signs = np.einsum('ij,ij->i', vec1s, vec2s)
            dist = np.abs(np.einsum('ij,ij->i', vec1s, line_norms))

            nearby_segments = []
            directions = []
            for j in np.arange(len(signs)):
                if signs[j] < 0.0:
                    if dist[j] < dist_threshold:
                        if len(directions) == 0:
                            directions.append(line_vecs[j])
                        else:
                            find_match = False
                            for dir_idx in range(0, len(directions)):
                                normalized_vec = directions[dir_idx] / np.linalg.norm(directions[dir_idx])
                                dot_value = np.dot(line_vecs[j], normalized_vec)
                                if abs(dot_value) > 0.91:
                                    find_match = True
                                    break
                            if not find_match:
                                directions.append(line_vecs[j])

            normalized_dirs = []
            for ind in range(0, len(directions)):
                vec = directions[ind] / np.linalg.norm(directions[ind])
                normalized_dirs.append(vec)

            point_directions.append(normalized_dirs)

           
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
        print "end computing"
        
        with open(sys.argv[3], 'wb') as fout:
            cPickle.dump(sample_point_cloud, fout, protocol=2)

        with open(sys.argv[4], 'wb') as fout:
            cPickle.dump(revised_canonical_directions, fout, protocol=2)
        return
    else:
        with open(sys.argv[3], 'rb') as fin:
            sample_point_cloud = cPickle.load(fin)

        with open(sys.argv[4], 'rb') as fin:
            revised_canonical_directions = cPickle.load(fin)

    visualize_sample_point_cloud(sample_point_cloud, 
                                 revised_canonical_directions,
                                 point_cloud, LOC, R)

    #track_idx = 0
    #arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    ##track = tracks[track_idx]
    #fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(111, aspect='equal')
    #for i in range(0, sample_point_cloud.locations.shape[0]):
    #    for direction in revised_canonical_directions[i]:
    #        ax.arrow(sample_point_cloud.locations[i][0],
    #                 sample_point_cloud.locations[i][1],
    #                 20*direction[0], 
    #                 20*direction[1],
    #                 width=0.5, head_width=5, fc='gray', ec='gray',
    #                 head_length=10, overhang=0.5, **arrow_params)
  
    #plt.show()

    return

    count = 0
    track = tracks[track_idx]
    query_distance = 50  # in meter
  
    clusters = []
    count = 0
   
    compute_sample_cluster = False
    if compute_sample_cluster:
        sample_clusters = []
        for track_idx in range(0, 1000):
            track = tracks[track_idx]
            for pt_idx in range(0, len(track.utm)):
                if len(track.utm) <= 1:
                    continue
                pt = np.array((track.utm[pt_idx][0], track.utm[pt_idx][1]))
                #ax.plot(pt[0], pt[1], 'or')
                #if pt_idx < len(track.utm) - 1:
                #    u = track.utm[pt_idx+1][0]-pt[0]
                #    v = track.utm[pt_idx+1][1]-pt[1]
                #    if abs(u) + abs(v) > 2:
                #        ax.arrow(pt[0], pt[1], u,
                #                 v, width=0.5, head_width=5, fc='r', ec='r',
                #                 head_length=10, overhang=0.5, **arrow_params)

                in_dir = np.array([0.0, 0.0])
                out_dir = np.array([0.0, 0.0])
                if pt_idx < len(track.utm) - 1:
                    out_dir = np.array((track.utm[pt_idx+1][0]-pt[0], track.utm[pt_idx+1][1]-pt[1]))
                    vec_norm = np.linalg.norm(out_dir)
                    if vec_norm < 1:
                        out_dir = np.array([0.0, 0.0])
                    else:
                        out_dir /= vec_norm
                if pt_idx >= 1:
                    in_dir = np.array((pt[0]-track.utm[pt_idx-1][0], pt[1]-track.utm[pt_idx-1][1]))
                    vec_norm = np.linalg.norm(in_dir)
                    if vec_norm < 1:
                        in_dir = np.array([0.0, 0.0])
                    else:
                        in_dir /= vec_norm

                # search nearby sample points
                neighbor_idxs = sample_point_kdtree.query_ball_point(pt, query_distance)
                # Filter sample by angle 
                filtered_sample_by_angle = []
                filtered_sample_by_angle_directions = []
                for sample_idx in neighbor_idxs:
                    for direction in revised_canonical_directions[sample_idx]:
                        if np.dot(direction, in_dir) >= 0.8:
                            filtered_sample_by_angle.append(sample_idx)
                            filtered_sample_by_angle_directions.append(np.copy(direction))
                            break
                        if np.dot(direction, out_dir) >= 0.8:
                            filtered_sample_by_angle.append(sample_idx)
                            filtered_sample_by_angle_directions.append(np.copy(direction))
                            break
                # Filter sample by distance
                filtered_sample = []
                filtered_sample_directions = []
                pt_in_norm = np.array([-1*in_dir[1], in_dir[0]])
                pt_out_norm = np.array([-1*out_dir[1], out_dir[0]])
                for s in range(0, len(filtered_sample_by_angle)):
                    sample_idx = filtered_sample_by_angle[s]
                    vec = sample_point_cloud.locations[sample_idx] - pt
                    if abs(np.dot(vec, pt_in_norm)) < query_distance*0.4:
                        filtered_sample.append(sample_idx)
                        filtered_sample_directions.append(filtered_sample_by_angle_directions[s])
                        continue
                    if abs(np.dot(vec, pt_out_norm)) < query_distance*0.4:
                        filtered_sample.append(sample_idx)
                        filtered_sample_directions.append(filtered_sample_by_angle_directions[s])
                        continue
                
                if len(filtered_sample) == 0:
                    continue

                new_cluster = SampleCluster(filtered_sample, 
                                            filtered_sample_directions, 
                                            sample_point_cloud) 

                # Check with existing clusters
                found_merge = False
                for cluster in sample_clusters:
                    similarity = cluster.compute_similarity(new_cluster)
                    if similarity >= 0.5:
                        cluster.merge_cluster(new_cluster)
                        found_merge = True
                        break
                if not found_merge:
                    sample_clusters.append(new_cluster)

            #for sample_idx in filtered_sample:
            #    ax.plot(sample_point_cloud.locations[sample_idx][0],
            #            sample_point_cloud.locations[sample_idx][1],
            #            '.', color=const.colors[pt_idx%7])
        with open(sys.argv[4], 'wb') as fout:
            cPickle.dump(sample_clusters, fout, protocol=2)
        return
    else:
        with open(sys.argv[4], 'rb') as fin:
            sample_clusters = cPickle.load(fin)

    # cluster sample_clusters
    compute_dbscan = False
    if compute_dbscan:
        N = len(sample_clusters)
        distance_matrix = np.zeros((N, N))
        for i in range(0, N):
            for j in range(i+1, N):
                cluster1 = sample_clusters[i]
                cluster2 = sample_clusters[j]
                similarity = cluster1.compute_similarity(cluster2)
                if similarity < 1e-3:
                    similarity = 1e-3
                distance_matrix[i,j] = 1.0 / similarity
                distance_matrix[j,i] = 1.0 / similarity
        print "max=",np.amax(distance_matrix)
        print "min=",np.amin(distance_matrix)
        print "DBSCAN started."
        t_start = time.time()
        db = DBSCAN(eps=2, min_samples=1, metric='precomputed').fit(distance_matrix)
        print "DBSCAN took %d sec."%(int(time.time() - t_start))
        
        core_samples = db.core_sample_indices_
        labels = db.labels_
        n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        print "There are %d clusters."%n_cluster
        unique_labels = set(labels)
        new_clusters = []
        for k in unique_labels:
            if k==-1:
                continue
            class_members = [index[0] for index in np.argwhere(labels==k)]
            starting_cluster = sample_clusters[class_members[0]]
            for j in range(1, len(class_members)):
                starting_cluster.merge_cluster(sample_clusters[class_members[j]])
            new_clusters.append(starting_cluster)

        with open(sys.argv[5], "wb") as fout:
            cPickle.dump(new_clusters, fout, protocol=2)
        return
    else:
        with open(sys.argv[5], "rb") as fin:
            new_clusters = cPickle.load(fin)

    for cluster_idx in range(0, len(new_clusters)):
        cluster = new_clusters[cluster_idx]
        color = const.colors[cluster_idx%7]
        ax.plot(sample_point_cloud.locations[cluster.member_samples.keys(), 0],
                sample_point_cloud.locations[cluster.member_samples.keys(), 1],
                '.', color=color)
        ax.plot(cluster.mass_center[0],
                cluster.mass_center[1],
                'o', color=color)
        if np.linalg.norm(cluster.direction) > 0.1:
            ax.arrow(cluster.mass_center[0],
                     cluster.mass_center[1],
                     100*cluster.direction[0],
                     100*cluster.direction[1],
                     width=3, head_width=20, fc=color, ec=color,
                     head_length=20, overhang=0.5, **arrow_params)
    #for track in tracks:
    #    count += 1
    #    if count == 10:
    #        break
    #    ax.plot([pt[0] for pt in track.utm],
    #            [pt[1] for pt in track.utm],
    #            'r.', markersize=12)
    #    for i in range(1, len(track.utm)):
    #        vec_e = track.utm[i][0] - track.utm[i-1][0]
    #        vec_n = track.utm[i][1] - track.utm[i-1][1]
    #        if abs(vec_e) + abs(vec_n) < 1.0:
    #            continue
    #        ax.arrow(track.utm[i-1][0], track.utm[i-1][1],
    #                 vec_e, vec_n,
    #                 width=1, head_width=10, fc='b', ec='b',
    #                 head_length=20, overhang=0.5, **arrow_params)
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    print "# clusters:",len(sample_clusters)
    plt.show()
    return

    for pt in track.utm:
        if pt[0]>=LOC[0]-R and pt[0]<=LOC[0]+R \
            and pt[1]>=LOC[1]-R and pt[1]<=LOC[1]+R:
            # Search point
            dist, nearby_idx = point_cloud_kdtree.query(np.array([pt[0], pt[1]]))
            for j in range(0, len(point_directions[nearby_idx])):
                direction = point_directions[nearby_idx][j]
                ax.plot(pt[0], pt[1], 'or')
                ax.arrow(point_cloud.locations[nearby_idx][0],
                         point_cloud.locations[nearby_idx][1],
                         20*direction[0], 20*direction[1], fc='r', ec='r',
                         width=0.5, head_width=5,
                         head_length=10, overhang=0.5, **arrow_params)
                if np.linalg.norm(point_cloud.directions[nearby_idx]) > 0.1:
                    ax.arrow(point_cloud.locations[nearby_idx][0],
                             point_cloud.locations[nearby_idx][1],
                             20*point_cloud.directions[nearby_idx][0], 
                             20*point_cloud.directions[nearby_idx][1], 
                             width=0.5, head_width=5, fc='b', ec='b',
                             head_length=10, overhang=0.5, **arrow_params)

    #ax.set_xlim([LOC[0]-R, LOC[0]+R])
    #ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show() 
    return
    #track = tracks[track_idx]
    #count = 0
    #for pt in track.utm:
    #    if pt[0]>=LOC[0]-R and pt[0]<=LOC[0]+R \
    #        and pt[1]>=LOC[1]-R and pt[1]<=LOC[1]+R:
    #        count += 1
    #        # search nearby lines
    #        for line in lines:
    #            vec = np.array((line[1][0]-line[0][0], line[1][1]-line[0][1]))
    #            vec_norm = np.linalg.norm(vec)
    #            vec /= vec_norm
    #            vec1 = np.array((pt[0]-line[0][0], pt[1]-line[0][1]))
    #            vec2 = np.array((pt[0]-line[1][0], pt[1]-line[1][1]))
    #            if np.dot(vec1, vec2) < 0:
    #                norm_vec = np.array((-1.0*vec[1], vec[0]))
    #                dist = abs(np.dot(vec1, norm_vec))
    #                if dist < 10.0:
    #                    ax.plot([line[0][0], line[1][0]],
    #                            [line[0][1], line[1][1]],
    #                            '-', color=const.colors[count%7])

    ax.plot([pt[0] for pt in track.utm],
            [pt[1] for pt in track.utm],
            '.-r', linewidth=3)
    ax.plot(track.utm[0][0],
            track.utm[0][1],
            'or')

    #for line in lines:
    #    ax.plot([line[0][0], line[1][0]],
    #            [line[0][1], line[1][1]],
    #            'r-')

    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])

    plt.show()

    return

if __name__ == "__main__":
    sys.exit(main())
