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

from l1 import l1
from cvxopt import matrix

from point_cloud import PointCloud
from road_segment import RoadSegment

import const

def extract_road_segments(sample_point_cloud,
                          lines,
                          search_radius = 25,
                          angle_threshold = np.pi/8.0):
    """ Extract road segments based on RanSac.
        Args:
            - sample_point_cloud: PointCloud object
            - lines: a list of line segments
            - search_radius: in meters, searching range at each sample location
            - angle_threshold
        Return:
            - road_segments: a list of RoadSegment objects
    """
    road_segments = []

    # Sample kdtree
    sample_kdtree = spatial.cKDTree(sample_point_cloud.locations)
    sample_marked = np.zeros(sample_point_cloud.locations.shape[0])
    line_lengths = []
    for line in lines:
        line_length = np.linalg.norm(line[1]-line[0])
        line_lengths.append(-1*line_length)
    
    sorted_idxs = np.argsort(line_lengths)
    line_marked = [False]*len(lines)
    count = 0
    while True:
        # Find the longest unmarked line segments
        for i in range(0, len(sorted_idxs)):
            line_idx = sorted_idxs[i]
            if not line_marked[line_idx]:
                break
        if line_marked[line_idx]:
            break

        selected_line = lines[line_idx]
        line_marked[line_idx] = True
        # Find nearby sample points
        nearby_idxs = []
        line_length = np.linalg.norm(selected_line[1]-selected_line[0])
        n_pt_to_insert = max(3, int(1.5*line_length / search_radius))
        pt_xs = np.linspace(selected_line[0][0], selected_line[1][0], n_pt_to_insert)
        pt_ys = np.linspace(selected_line[0][1], selected_line[1][1], n_pt_to_insert)
        for i in range(0, n_pt_to_insert):
            nb_idxs = sample_kdtree.query_ball_point(np.array([pt_xs[i], pt_ys[i]]), search_radius)
            nearby_idxs.extend(nb_idxs)
        
        directions = [] 
        direction = selected_line[1] - selected_line[0]
        direction /= np.linalg.norm(direction)
        directions.append(direction)
        directions.append(-1*direction)
        
        for direction in directions:
            consensus_set = []
            for pt_idx in nearby_idxs:
                if sample_marked[pt_idx]:
                    continue
                if np.dot(direction, sample_point_cloud.directions[pt_idx]) > np.cos(angle_threshold):
                    consensus_set.append(pt_idx)
            if len(consensus_set) < 5:
                continue
            
            consensus_pts = np.copy(sample_point_cloud.locations[consensus_set, :])
            mean_vec = np.mean(consensus_pts, axis=0)
           
            normalized_pts = consensus_pts - mean_vec
            M = np.dot(normalized_pts.T, normalized_pts) / normalized_pts.shape[0]
            u, s, v = np.linalg.svd(M)
            box_dir = u[0,:]
            if np.dot(direction, box_dir) < 0:
                box_dir *= -1
           
            # Check if the road segment is of good quality
            box_norm_dir = np.array([-1*box_dir[1], box_dir[0]]) 
            length_dir_proj = []
            norm_dir_proj = []
            for idx in consensus_set:
                vec = sample_point_cloud.locations[idx] - mean_vec
                length_dir_proj.append(np.dot(vec, box_dir))
                norm_dir_proj.append(np.dot(vec, box_norm_dir))
            length_dir_proj = np.array(length_dir_proj)
            norm_dir_proj = np.array(norm_dir_proj)
            max_length_proj = max(length_dir_proj)
            min_length_proj = min(length_dir_proj)

            box_center = mean_vec + box_dir*0.5*(max_length_proj+min_length_proj)
            abs_norm_dir_proj = np.abs(norm_dir_proj)
            abs_norm_dir_proj.sort()
            
            box_half_length = 0.5*(max_length_proj - min_length_proj)
            box_half_width = abs_norm_dir_proj[int(0.9*len(abs_norm_dir_proj))]
                # Check Gap
            length_dir_proj.sort()
            max_gap = -np.inf
            for idx in range(1, len(length_dir_proj)):
                gap = length_dir_proj[idx] - length_dir_proj[idx-1]
                if gap > max_gap:
                    max_gap = gap
            if max_gap > 20:
                continue
            if box_half_width > 0:
                ratio = box_half_length / box_half_width
            else:
                ratio = 0
            if ratio < 2.0:
                # Aspect ratio too small
                continue
            if box_half_length*box_half_width < 100:
                # Area too small
                continue
            
            total_length = max_length_proj - min_length_proj
            if total_length < 0.7*line_length:
                # Shouldn't fit such a long line
                continue

            segment = RoadSegment(box_center, box_dir, box_norm_dir, box_half_length, box_half_width)
            # Remove consensus set points
            for idx in consensus_set:
                sample_marked[idx] = True
                
            road_segments.append(segment)
   
    return road_segments

   #while True:
   #     count += 1
   #     print "Now at %d:"%count
   #     print "\t%d segments generated"%(len(road_segments))
   #     if count == 150:
   #         break
   #      
   #     # Randomly pick an unmarked sample point as box center
   #     unmarked_sample_idxs = np.where(sample_marked == 0)[0]
   #     visited_samples = np.zeros([len(unmarked_sample_idxs)])
   #     print "\t#unmarked: ",len(unmarked_sample_idxs)
   #     if len(unmarked_sample_idxs) == 0:
   #         break
   #     
   #     n_candidate_sample = 0
   #     best_candidate_consensus_set = []
   #     best_candidate_score = 0
   #     inner_iter = 0
   #     # Generate the best road segments from a set of <= 16 candidates
   #     while n_candidate_sample < N_SAMPLE_PER_STEP and inner_iter < 100:
   #         inner_iter += 1
   #         candidate_pool = np.where(visited_samples == 0)[0]
   #         if len(candidate_pool) == 0:
   #             break
   #         # Randomly pick an unvisited unmarked point
   #         tmp_idx = np.random.randint(0, len(candidate_pool))
   #         start_idx = unmarked_sample_idxs[candidate_pool[tmp_idx]]
   #         visited_samples[candidate_pool[tmp_idx]] = 1
   #         box_dir = sample_point_cloud.directions[start_idx]
   #         if np.linalg.norm(box_dir) < 0.1:
   #             continue

   #         box_center = sample_point_cloud.locations[start_idx]
   #         box_norm = np.array([-1*box_dir[1], box_dir[0]])
   #         box_half_length = search_radius
   #         box_half_width = search_radius
   #    
   #         iter_count = 0
   #         while True:
   #             iter_count += 1
   #             # Search nearby points
   #             nearby_idxs = []
   #             n_pt = int(2*box_half_length/search_radius+0.5) + 1
   #             vec_xs = np.linspace(box_center[0]-box_half_length*box_dir[0],
   #                                  box_center[0]+box_half_length*box_dir[0])
   #             vec_ys = np.linspace(box_center[1]-box_half_length*box_dir[1],
   #                                  box_center[1]+box_half_length*box_dir[1])

   #             interpolated_pts = np.array((vec_xs, vec_ys)).T
   #             for pt in interpolated_pts:
   #                 nb_idxs = sample_kdtree.query_ball_point(pt, search_radius)
   #                 nearby_idxs.extend(nb_idxs)
   #             nearby_idxs = set(nearby_idxs) 

   #             consensus_idxs = []
   #             # Expand consensus set
   #             for i in nearby_idxs:
   #                 if sample_marked[i] >= 1:
   #                     continue
   #                 # Check direction compatibility
   #                 if np.dot(sample_point_cloud.directions[i], box_dir) < np.cos(angle_threshold):
   #                     if np.linalg.norm(sample_point_cloud.directions[i]) > 0.1:
   #                         continue
   #                 consensus_idxs.append(i)
   #           
   #             if len(consensus_idxs) < 3:
   #                 break

   #             # Compute new box parameters
   #                 # - Compute new direction
   #             consensus_pts = np.copy(sample_point_cloud.locations[consensus_idxs, :])
   #             sum_vec = np.array([0.0, 0.0])
   #             for pt in consensus_pts:
   #                 sum_vec += pt

   #             mean_vec = sum_vec / len(consensus_pts)
   #             consensus_pts -= mean_vec
   #             M = np.dot(consensus_pts.T, consensus_pts) / consensus_pts.shape[0]
   #             u, s, v = np.linalg.svd(M)
   #             new_dir = u[0,:]
   #             if np.dot(new_dir, box_dir) < 0:
   #                 new_dir *= -1
   #           
   #             alpha = 0.2
   #             new_dir = new_dir*(1-alpha) + sample_point_cloud.directions[start_idx]
   #             new_dir /= np.linalg.norm(new_dir)
   #             if np.dot(new_dir, sample_point_cloud.directions[start_idx]) < np.cos(angle_threshold):
   #                 break

   #             new_norm_dir = np.array([-1.0*new_dir[1], new_dir[0]]) 
   #                 # - Compute new center
   #             new_center = mean_vec
   #                 # - Compute width and length 
   #             length_dir_proj = []
   #             norm_dir_proj = []
   #             for idx in consensus_idxs:
   #                 vec = sample_point_cloud.locations[idx] - new_center
   #                 length_proj = np.dot(vec, new_dir)
   #                 norm_proj = np.dot(vec, new_norm_dir)
   #                 length_dir_proj.append(length_proj)
   #                 norm_dir_proj.append(norm_proj)

   #             length_dir_proj = np.array(length_dir_proj)
   #             norm_dir_proj = np.array(norm_dir_proj)

   #             new_box_half_length = 0.45*(max(length_dir_proj) - min(length_dir_proj))
   #             new_box_half_width = 1.2*np.sqrt(s[1])
   #             
   #             if new_box_half_length < 1.0:
   #                 break
   #             
   #             if new_box_half_width > search_radius:
   #                 new_box_half_width = search_radius

   #             length_dir_proj.sort()
   #             max_gap = -np.inf
   #             for idx in range(1, len(length_dir_proj)):
   #                 gap = length_dir_proj[idx] - length_dir_proj[idx-1]
   #                 if gap > max_gap:
   #                     max_gap = gap
   #             if max_gap > 20:
   #                 break

   #             delta_center = np.linalg.norm(new_center - box_center)
   #             delta_half_length = new_box_half_length - box_half_length
   #             delta_half_width = new_box_half_width - box_half_width

   #             box_center = new_center
   #             box_dir = new_dir
   #             box_norm = new_norm_dir
   #             box_half_length = new_box_half_length
   #             box_half_width = new_box_half_width

   #             if delta_center < 1.0 and delta_half_length < 1.0 and delta_half_width < 1.0:
   #                 break
   #             
   #             if iter_count >= 50:
   #                 break
   #        
   #         if box_half_width > 0.01:
   #             aspect_ratio = box_half_length / box_half_width
   #         else:
   #             aspect_ratio = 0.0
   #             continue
   #     
   #         # Compute new consensus set
   #         nearby_idxs = []
   #         n_pt = int(2*box_half_length/search_radius+0.5) + 1
   #         vec_xs = np.linspace(box_center[0]-box_half_length*box_dir[0],
   #                              box_center[0]+box_half_length*box_dir[0])
   #         vec_ys = np.linspace(box_center[1]-box_half_length*box_dir[1],
   #                              box_center[1]+box_half_length*box_dir[1])

   #         interpolated_pts = np.array((vec_xs, vec_ys)).T
   #         for pt in interpolated_pts:
   #             nb_idxs = sample_kdtree.query_ball_point(pt, search_radius)
   #             nearby_idxs.extend(nb_idxs)
   #         nearby_idxs = set(nearby_idxs) 

   #         consensus_idxs = []
   #         length_dir_proj = []
   #         norm_dir_proj = []
   #         # Expand consensus set
   #         for i in nearby_idxs:
   #             if sample_marked[i] >= 1:
   #                 continue
   #             # Check direction compatibility
   #             if np.dot(sample_point_cloud.directions[i], box_dir) < np.cos(angle_threshold):
   #                 if np.linalg.norm(sample_point_cloud.directions[i]) > 0.1:
   #                     continue
   #             consensus_idxs.append(i)
   #             vec = sample_point_cloud.locations[i] - box_center
   #             vec_x = np.dot(vec, box_dir)
   #             vec_y = np.dot(vec, box_norm)
   #             if abs(vec_x) <= box_half_length\
   #                     and abs(vec_y) <= box_half_width:
   #                 consensus_idxs.append(i) 
   #                 length_dir_proj.append(vec_x)
   #                 norm_dir_proj.append(vec_y)
   #         
   #         # Check gap
   #         length_dir_proj.sort()
   #         max_gap = -np.inf
   #         for idx in range(1, len(length_dir_proj)):
   #             gap = length_dir_proj[idx] - length_dir_proj[idx-1]
   #             if gap > max_gap:
   #                 max_gap = gap
   #         if max_gap > 20:
   #             continue

   #         if aspect_ratio < 3.0 or len(consensus_idxs) < 5:
   #             continue
   #         
   #         # Compute box score
   #         if box_half_width*box_half_length <= 200:
   #             continue

   #         # Divide the sub-boxes
   #         box_size = 10
   #         n_l = max(1, int(2*box_half_length/box_size+0.5))
   #         n_w = max(1, int(2*box_half_width/box_size+0.5))
   #         n_subbox = n_l*n_w
   #         subbox_pt_count = np.zeros(n_subbox)
   #         delta_l = 2*box_half_length/n_l
   #         delta_w = 2*box_half_width/n_w
   #         bottom_left_corner = box_center - box_half_length*box_dir - box_half_width*box_norm

   #         # Angle difference sum
   #         sum_angle_diff = 0.0
   #         for pt_idx in consensus_idxs:
   #             pt_loc = sample_point_cloud.locations[pt_idx]
   #             vec = pt_loc - bottom_left_corner
   #             proj_i = np.dot(vec, box_dir)
   #             proj_j = np.dot(vec, box_norm)
   #             pt_i = int(proj_i/delta_l)
   #             pt_j = int(proj_j/delta_w)
   #             if pt_i<0 or pt_i>=n_l or pt_j<0 or pt_j>n_w:
   #                 continue
   #             pt_bin_id = pt_j*n_l + pt_i
   #             if pt_bin_id<0 or pt_bin_id>= n_subbox:
   #                 continue
   #             subbox_pt_count[pt_bin_id] += 1.0

   #             box_dir_3d = np.array([box_dir[0], box_dir[1], 0.0])
   #             sample_dir = sample_point_cloud.directions[pt_idx]
   #             sample_dir_3d = np.array([sample_dir[0], sample_dir[1], 0.0])
   #             angle_diff = np.arcsin(np.dot(np.cross(box_dir_3d, sample_dir_3d), np.array([0.0,0.0,1.0])))
   #             sum_angle_diff += abs(angle_diff)
   #             
   #             vec = pt_loc - box_center
   #             dir_proj = np.dot(vec, box_dir)
   #             norm_proj = np.dot(vec, box_norm)
   #         sum_angle_diff = max(1e-3, abs(sum_angle_diff))
   #         score = len(consensus_idxs) * box_half_length / sum_angle_diff

   #         neary_empty_subbox_count = 0
   #         for s in range(0, n_subbox):
   #             if subbox_pt_count[s] == 0:
   #                 neary_empty_subbox_count += 1
   #         empty_ratio = float(neary_empty_subbox_count)/len(subbox_pt_count)
   #         if empty_ratio > 0.5:
   #             continue

   #         score *= (1-empty_ratio)
   #         if score > best_candidate_score:
   #             best_candidate_score = score
   #             best_candidate = RoadSegment(box_center, box_dir, box_norm, box_half_length, box_half_width)
   #             best_candidate_consensus_set = consensus_idxs
   #         n_candidate_sample += 1 

   #     if n_candidate_sample > 0:
   #         road_segments.append(best_candidate)
   #         for idx in best_candidate_consensus_set:
   #             sample_marked[idx] += 1

   # return road_segments

def main():
    parser = OptionParser()
    parser.add_option("-s", "--sample_point_cloud", dest="sample_point_cloud", help="Input sample point cloud filename", metavar="SAMPLE_POINT_CLOUD", type="string")
    parser.add_option("-l", "--lines", dest="extracted_lines", help="Input extracted line segments", metavar="LINE_SEGMENTS", type="string")
    parser.add_option("-r", "--road_segment", dest="road_segment", help="Output road segment filename", metavar="ROAD_SEGMENT", type="string")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)
    (options, args) = parser.parse_args()

    if not options.sample_point_cloud:
        parser.error("Input sample_point_cloud filename not found!")
    if not options.extracted_lines:
        parser.error("Input extracted lines file not found!")
    if not options.road_segment:
        parser.error("Output road_segment filename not specified!")

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

    with open(options.extracted_lines, 'rb') as fin:
        lines = cPickle.load(fin)

    # Generate Road Segments
    road_segments = extract_road_segments(sample_point_cloud, lines)

    print "There are %d boxes."%len(road_segments)
    #with open(sys.argv[3], 'wb') as fout:
    #    cPickle.dump(road_segments, fout, protocol=2)

    # Visualization
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(sample_point_cloud.locations[:,0], 
            sample_point_cloud.locations[:,1], 
            '.', color='gray')

    for i in range(0, len(road_segments)):
        box_center = road_segments[i].center
        box_half_width = road_segments[i].half_width
        box_half_length = road_segments[i].half_length
        
        box_dir = road_segments[i].direction
        box_norm = road_segments[i].norm_dir

        p0 = box_center - box_half_length*box_dir + box_half_width*box_norm
        p1 = box_center + box_half_length*box_dir + box_half_width*box_norm
        p2 = box_center + box_half_length*box_dir - box_half_width*box_norm
        p3 = box_center - box_half_length*box_dir - box_half_width*box_norm
        arrow_p0 = box_center - box_half_length*box_dir
        arrow_p1 = box_center + box_half_length*box_dir
        
        #ax.plot([arrow_p0[0], arrow_p1[0]], 
        #        [arrow_p0[1], arrow_p1[1]], '-', linewidth=2)
        color = const.colors[i%7]
        ax.plot([p0[0], p1[0]], [p0[1],p1[1]], '-', color=color)
        ax.plot([p1[0], p2[0]], [p1[1],p2[1]], '-', color=color)
        ax.plot([p2[0], p3[0]], [p2[1],p3[1]], '-', color=color)
        ax.plot([p3[0], p0[0]], [p3[1],p0[1]], '-', color=color)
        ax.arrow(arrow_p0[0],
                 arrow_p0[1],
                 2*box_half_length*box_dir[0],
                 2*box_half_length*box_dir[1],
                 width=2, head_width=10, fc=color, ec=color,
                 head_length=20, overhang=0.5, **arrow_params)
    
    ax.set_xlim([LOC[0]-R, LOC[0]+R])
    ax.set_ylim([LOC[1]-R, LOC[1]+R])
    plt.show()
    return

if __name__ == "__main__":
    sys.exit(main())
