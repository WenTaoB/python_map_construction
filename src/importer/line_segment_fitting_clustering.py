#!/usr/bin/env python
"""
Created on Mon Oct 29, 2013

@author: ChenChen
"""

import sys
import random
import cPickle
import copy
import time

from scipy import weave
from scipy.weave import converters
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm as CM
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics

import gps_track
import douglas_peucker_track_segmentation
import const

class LinearCurveModel:
    """
        Using Linear line segments to fit data.
    """
    def __init__(self):
        return
    
    def fit(self, data):
        """
            Fit the model using data.
                Args:
                    - data: the data points to fit;
                Return:
                    - model parameters.
        """
        u, s, v = np.linalg.svd(data, full_matrices=False)
        return v[2,:]

    def get_error(self, data, model):
        error = []

        delta_f = np.array([model[0], model[1]])
        delta_f_norm = np.linalg.norm(delta_f)
        for pt in data:
            tmp_error = np.abs(model[0]*pt[0] + model[1]*pt[1] + model[2])
            tmp_error /= delta_f_norm
            error.append(tmp_error)
        
        err_per_point = np.array(error)

        return err_per_point

    def visualize(self, data, model):
        line_norm = np.array([model[0], model[1]])
        line_norm /= np.linalg.norm(line_norm)

        line_dir = np.array([-1*line_norm[1], line_norm[0]])
        k = -1*model[2] / (model[0]*line_norm[0] + model[1]*line_norm[1])
        xc = k*line_norm

        modified_x = data[:,0] - xc[0]
        modified_y = data[:,1] - xc[1]
        modified_closeby_data = np.array([modified_x, modified_y]).T
        data_projection = np.dot(modified_closeby_data, line_dir)
        
        start_loc = np.min(data_projection)*line_dir + xc
        end_loc = np.max(data_projection)*line_dir + xc

        errors = np.dot(np.array((modified_x, modified_y)).T, line_norm)
        #error = np.linalg.norm(errors)
        error = np.max(np.abs(errors))

        return start_loc, end_loc, error 

def ransac_line_fitting(data, init_guess, k, t, d):
    """ RANSAC Line fitting with initial guess
    """
    linear_model = LinerCurveModel()

    dist_to_line = np.abs(np.dot(data, np.array([init_guess[0], init_guess[1]])) + init_guess[2])
    all_idxs = np.arange(data.shape[0])
    closeby_data_idxs = all_idxs[dist_to_line < t]
    closeby_data = data[closeby_data_idxs, :]

    if closeby_data.shape[0] < d:
        return []

    n = int(0.1*closeby_data.shape[0])
    if n > 100:
        n = 100

    iteration = 0
    best_model = []
    best_consensus_set = []
    best_error = np.inf
    while iteration < k:
        all_idxs = np.arange(closeby_data.shape[0])
        np.random.shuffle(all_idxs)

        maybe_inliers_idxs = all_idxs[:n]
        test_idxs = all_idxs[n:]

        maybe_inliers = closeby_data[maybe_inliers_idxs, :]
        maybe_model = linear_model.fit(maybe_inliers)

        test_data = closeby_data[test_idxs, :]

        test_err = linear_model.get_error(test_data, maybe_model)
        also_idxs = test_idxs[test_err < t]
        also_inliers = closeby_data[also_idxs, :]

        better_data = np.concatenate((maybe_inliers, also_inliers))
        better_model = linear_model.fit(better_data)

        if better_data.shape[0] > d:
            this_error = sum(linear_model.get_error(better_data, better_model))
            this_error /= better_data.shape[0]
            if this_error < best_error:
                # Update model
                best_model = better_model
                best_error = this_error
                
                # Recompute nearby points
                dist_to_line = linear_model.get_error(data, best_model)
                all_idxs = np.arange(data.shape[0])
                best_consensus_set = all_idxs[dist_to_line < t]
                closeby_data = data[best_consensus_set, :]

        iteration += 1
    return best_model, best_consensus_set

def check_peak(data, 
               line_param, 
               t, 
               min_seg_length, 
               with_idx = False):
    """ Check the continuity
        Args:
            - data: ndarray of shape (m, 2);
            - line_param: (a,b,c) of a line.
            - t: distance threshold
            - min_seg_length: minimum segment length
            - with_idx: assigned data_index
        Returns:
            - True/False: if it is an effective line;
            - segments_loc: [(start_e, start_n, end_e, end_n), ...] for each segments;
            - data_idxs: OPTIONAL, data points that have been assigned to segments.
    """
    # Get nearby points
    dist_to_line = np.dot(data, np.array([line_param[0], line_param[1]])) + line_param[2]
    dist_to_line = np.abs(dist_to_line)
    dist_to_line /= np.linalg.norm([line_param[0], line_param[1]])

    line_norm = np.array([line_param[0], line_param[1]])
    line_norm /= np.linalg.norm(line_norm)

    all_idxs = np.arange(data.shape[0])
    closeby_data_idxs = all_idxs[dist_to_line < t]
    closeby_data = data[closeby_data_idxs, :]

    if len(closeby_data_idxs) < 10:
        if with_idx:
            return False, [], []
        else:
            return False, []

    line_dir = np.array([-1*line_norm[1], line_norm[0]])
    
    k = -1*line_param[2] / (line_param[0]*line_norm[0] + line_param[1]*line_norm[1])
    xc = k*line_norm
    
    modified_x = closeby_data[:,0] - xc[0]
    modified_y = closeby_data[:,1] - xc[1]
    modified_closeby_data = np.array([modified_x, modified_y]).T

    tmp_data_projection = np.dot(modified_closeby_data, line_dir)
    
    sorted_idx = np.argsort(tmp_data_projection)
    data_projection = tmp_data_projection[sorted_idx]
    
    segments = []
    segment_start_idx = 0
    segment_end_idx = -1
    for idx in range(1,len(data_projection)):
        delta_dist = data_projection[idx] - data_projection[idx-1]
        if delta_dist >= 50: # Maximum gap in meter!!!!
            # Start of a new segment
            segment_end_idx = idx-1
            seg_length = data_projection[segment_end_idx] - data_projection[segment_start_idx]
            if seg_length > min_seg_length:
                segments.append((segment_start_idx, segment_end_idx))
            segment_start_idx = idx

    segment_end_idx = len(data_projection) - 1
    seg_length = data_projection[segment_end_idx] - data_projection[segment_start_idx]
    if seg_length > min_seg_length:
        pass
        segments.append((segment_start_idx, segment_end_idx))

    if len(segments) == 0:
        if with_idx:
            return False, [], []
        else:
            return False, []
    
    # Convert segments into start/end point pairs
    segments_loc = []
    assigned_data_idx = []
    for p_idx in segments:
        start_idx = p_idx[0]
        end_idx = p_idx[1]
        orig_data_idxs = closeby_data_idxs[sorted_idx[start_idx:(end_idx+1)]]
        assigned_data_idx.extend(list(orig_data_idxs))

        start_loc = data_projection[start_idx]*line_dir + xc
        end_loc = data_projection[end_idx]*line_dir + xc
        segments_loc.append((start_loc[0], start_loc[1], end_loc[0], end_loc[1]))
    
    if with_idx:
        return True, segments_loc, assigned_data_idx
    else:
        return True, segments_loc

def hough_line_fitting(data, 
                       n_theta_bin, 
                       delta_dist, 
                       n_min,
                       min_seg_length,
                       k, t, d):
    """ Implementing probabilistic hough line fitting algorithm.
        Args:
            - data: input ndarray data;
            - n_theta_bin: number of angle bins;
            - delta_dist: dist will be binned using hash, delta_dist denoting the min rounding;
              For example, delta_dist = 100 means we will round dist to 100m;
            - k, t, d: for RANSAC
        Return:
            - line segments.
    """
    theta = np.linspace(0, 2*np.pi, n_theta_bin)
    segments = []
    seg_id = 1
    while True:
        print "Now at %d-th iteration."%seg_id
        seg_id += 1
        
        if seg_id > 100:
            break
        
        if data.shape[0] < 400:
            break

        # Keep sampling until one effective peak is found
        # minimum # of points to be counted as an effective peak
        n_min = int(data.shape[0] * 0.2) 
        if n_min > 100:
            n_min = 100
        current_max = 0
        dist_bins = {}
        all_idxs = np.arange(data.shape[0])
        np.random.shuffle(all_idxs)
        found_peak = False
        for i in all_idxs:
            picked_idx = all_idxs[i]
            r = data[picked_idx,0]*np.cos(theta) + data[picked_idx,1]*np.sin(theta)
            for idx in np.arange(n_theta_bin):
                rounded_value = int(r[idx] / delta_dist) * delta_dist
                if dist_bins.has_key(rounded_value):
                    dist_bins[rounded_value][idx] += 1
                    if current_max < dist_bins[rounded_value][idx]:
                        current_max = dist_bins[rounded_value][idx]
                        max_loc = (rounded_value, idx)
                else:
                    dist_bins[rounded_value] = np.zeros(n_theta_bin).astype(int)
                    dist_bins[rounded_value][idx] += 1
                    if current_max < 1:
                        current_max = 1
                        max_loc = (rounded_value, idx)

            if current_max >= n_min:
                # Find one acceptable bin
                found_peak = True
                break

        if not found_peak:
            break

        # Deal with the peak
        data_idx_to_remove = []
        angle = theta[max_loc[1]]
        init_guess = np.array([np.cos(angle), np.sin(angle), -1*max_loc[0]])
        linear_model = LinerCurveModel()
        # Check if there is possible line segment that can be fitted from this init_guess
        is_effective, seg_locs = check_peak(data, 
                                            init_guess, 
                                            t, 
                                            min_seg_length)
        
        if not is_effective:
            continue

        # Iterative RANSAC fitting with initial guess being the peak line
        line_param, fit_data_idxs = ransac_line_fitting(data, init_guess, k, t, d)

        if line_param.shape == 0:
            continue

        is_effective, seg_locs, data_idx = check_peak(data,
                                                      line_param,
                                                      t,
                                                      min_seg_length,
                                                      with_idx = True)
        data_idx_to_remove.extend(data_idx)
        segments.extend(seg_locs)

        # Remove fitted data
        if len(data_idx_to_remove) > 0:
            data = np.delete(data, data_idx_to_remove, 0)

    return segments

def n_point_sequence(tracks, n_point):
    """
        Generate n-point sequence from tracks.
        For example, track = p0p1p2...
            For n_point = 3, the sequences generated will be p0p1p2, p1p2p3,...
        Args:
            - tracks: a collection of GPS tracks;
            - n_point: an integer;
        Return:
            - sequences: a list of point sequences.
                        Format: [[p0,p1,p2], ...].
    """
    sequences = []
    sequence_from = []
    cur_track_idx = 0
    for track in tracks:
        for i in range(0, len(track.utm)-n_point+1):
            sequence = []
            j = i
            sequence.append(track.utm[j])
            j += 1
            last_recorded_i = i
            count = 1
            while count < n_point:
                delta_d = np.sqrt((track.utm[j][0] - track.utm[last_recorded_i][0])**2 + \
                                  (track.utm[j][1] - track.utm[last_recorded_i][1])**2)
                if delta_d > 5.0:
                    count += 1
                    last_recorded_i = j
                    sequence.append(track.utm[j])
                j += 1
                if j == len(track.utm) - 1:
                    break

            if len(sequence) == n_point:
                sequences.append(sequence)
                sequence_from.append(cur_track_idx)
        cur_track_idx += 1
   
    sequences = np.array(sequences)
    return sequences, sequence_from

def segment_similarity(seg1, seg2):
    """
        Distance between two line segments.
        Args:
            - seg1, seg2: two line segments of format: [p0,p1]
        Return:
            - similarity: float value in [0, 1.0].
    """
    DELTA = 100 # in meter
    vec1 = np.array((seg1[0][0] - seg2[0][0], seg1[0][1] - seg2[0][1]))
    vec2 = np.array((seg1[1][0] - seg2[1][0], seg1[1][1] - seg2[1][1]))
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    vec3 = np.array((seg1[1][0] - seg1[0][0], seg1[1][1] - seg1[1][0]))
    vec4 = np.array((seg2[1][0] - seg2[0][0], seg2[1][1] - seg2[1][0]))
    vec3_norm = np.linalg.norm(vec3)
    vec4_norm = np.linalg.norm(vec4)

    if vec3_norm < 1.0 or vec4_norm < 1.0:
        similarity = 0
        return similarity

    dot_product = np.dot(vec3, vec4) / vec3_norm / vec4_norm
    similarity = np.exp(-1*vec1_norm**2/DELTA**2)*\
                 np.exp(-1*vec2_norm**2/DELTA**2)*\
                 dot_product
    return similarity

def my_distance(a, b):
    N = int(len(a))
    code = """
        #include <math.h>
        float result = 0.0;
        for (int i=0; i<N; ++i){
            result += a(i)*b(i);
        }
        return_val = sqrt(fabs(result));
    """
    return weave.inline(code, ['N', 'a', 'b'], 
                        type_converters=converters.blitz,
                        compiler='gcc')

def line_segment_fitting(point_sequences, 
                         sequence_from, 
                         min_length,
                         max_error):
    linear_model = LinearCurveModel()
    fitted_segments = []
    segment_from = []
    fitting_errors = []
    for i in range(0, len(point_sequences)):
        if (i+1)%10000 == 0:
            print "\tNow at sequence %d."%(i+1)
        
        sequence = point_sequences[i]
        ones = np.ones(len(sequence))
        data = np.array((sequence[:,0], sequence[:,1], ones)).T
        line_model = linear_model.fit(data)
        p1,p2,fitting_error = linear_model.visualize(data, line_model)
        length = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        if length > min_length and fitting_error < max_error:
            fitted_segments.append((p1[0], p1[1], p2[0], p2[1]))
            segment_from.append(i)
            fitting_errors.append(fitting_error)

    segment_from = np.array(segment_from)
    fitted_segments = np.array(fitted_segments)
    fitting_errors = np.array(fitting_errors)

    return fitted_segments, segment_from, fitting_errors

def refined_fitting(segment_idxs, segment_from, sequences):
    linear_model = LinearCurveModel()
    point_collection = []
    for seg_idx in segment_idxs:
        sequence = sequences[segment_from[seg_idx]]
        for pt in sequence:
            point_collection.append((pt[0], pt[1], 1.0))
    data = np.array(point_collection)
    line_model = linear_model.fit(data)
    p1,p2,fitting_error = linear_model.visualize(data,line_model)
    return (p1[0], p1[1], p2[0], p2[1]), fitting_error

def show_segments(fitted_segments):
    color_strings = ['b', 'g', 'r', 'c', 'm', 'y']
    colors = []
    lines = [] 
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, aspect='equal')

    for i in range(0, len(fitted_segments)):
        seg = fitted_segments[i]
        lines.append([(seg[0], seg[1]),(seg[2], seg[3])])
        colors.append(color_strings[i%6])
    collection = LineCollection(lines, colors=colors, linewidths=2)
    ax.add_collection(collection)
    ax.set_xlim([const.SF_small_RANGE_SW[0], const.SF_small_RANGE_NE[0]])
    ax.set_ylim([const.SF_small_RANGE_SW[1], const.SF_small_RANGE_NE[1]])
    plt.show()

def main():
    with open(sys.argv[1], "r") as fin:
        tracks = cPickle.load(fin)
    print "%d tracks loaded."%len(tracks)

    # Generate n-point sequences
    point_sequences, sequence_from = \
                        douglas_peucker_track_segmentation.dp_segmentation(tracks, 25)
    print "There are %d sequences."%len(point_sequences)

    # Line segment fitting
    fitted_segments, segment_from, fitting_errors = \
                                line_segment_fitting(point_sequences,
                                                     sequence_from,
                                                     250,
                                                     10)
    print "%d segments fitted."%len(fitted_segments)

    # Visualize fitted segments
    #show_segments(fitted_segments)
    #return

    line_vecs = []
    line_norms = []
    segment_midpoints = []
    for segment in fitted_segments:
        line_vec = np.array((segment[2]-segment[0], segment[3]-segment[1]))
        segment_midpoints.append((0.5*(segment[2]+segment[0]), 0.5*(segment[1]+segment[3])))
        line_vec_length = np.linalg.norm(line_vec)
        if line_vec_length < 0.001:
            line_vec *= 0.0
        else:
            line_vec /= line_vec_length
        line_norms.append((-1*line_vec[1], line_vec[0]))
        line_vecs.append(line_vec)
    line_vecs = np.array(line_vecs)
    line_norms = np.array(line_norms)
    segment_midpoints = np.array(segment_midpoints)

    angle_distance = 1.1 - np.dot(line_vecs, line_vecs.T)

    N = len(fitted_segments)
    Y = np.zeros((N,N))

    for i in range(0, N):
        if i%1000 == 0:
            print "now at ",i
        vec1s = fitted_segments[:,0:2] - fitted_segments[i,0:2]
        vec2s = fitted_segments[:,0:2] - fitted_segments[i,2:4]
        dist = abs(np.dot(vec1s, line_norms[i]))
        signs = np.einsum('ij,ij->i', vec1s, vec2s)

        for j in range(0, N):
            if j != i:
                if signs[j] < 0:
                    Y[i,j] = dist[j] + 1
                    if Y[j,i] > Y[i,j]:
                        Y[j,i] = Y[i,j]
                else:
                    Y[i,j] = np.inf
    distance_matrix = angle_distance*Y

    #N = len(fitted_segments)
    #dist_matrix = np.zeros((N,N)) 
    #for i in range(0, N):
    #    if i%1000 == 0:
    #        print "now at ",i
    #    dist_matrix[i,i] = 0
    #    for j in range(i+1, N):
    #        dist_matrix[i,j] = my_distance(fitted_segments[i], fitted_segments[j])
    #        dist_matrix[j,i] = dist_matrix[i,j]
    #t_end = time.time()
    #print "it tooks %d sec."%(int(t_end-t_start))
    #return

    print "DBSCAN started."
    t_start = time.time()
    db = DBSCAN(eps=0.3, min_samples=3, metric='precomputed').fit(distance_matrix)
    t_end = time.time()
    print "DBSCAN took %d sec."%(int(t_end - t_start))

    core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    fig = plt.figure(figsize=(9, 9))
    #ax1 = fig.add_subplot(111, aspect='equal')
    ax2 = fig.add_subplot(111, aspect='equal') 

    unique_labels = set(labels)
    print('Number of clusters: %d' % len(unique_labels))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    rand_labels = list(unique_labels)
    random.shuffle(rand_labels)
   
    count = 0
    for k, col in zip(rand_labels, colors):
        count += 1
        linewidth = 2
        if k == -1:
            # Black for noise
            col = 'k'
            linewidth = 0.1
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples if labels[index] == k]
        lines = []
        new_fitted, error = refined_fitting(class_members, segment_from, point_sequences)
        for index in class_members:
            seg = fitted_segments[index]
            lines.append([(seg[0], seg[1]),(seg[2], seg[3])])
        collection = LineCollection(lines, colors=col, linewidths=linewidth)
        if error < 20:
            ax2.plot([new_fitted[0], new_fitted[2]],
                     [new_fitted[1], new_fitted[3]],
                     '-', linewidth=2)
            #pass
        #ax1.add_collection(collection)

    #ax1.set_xlim([const.SF_small_RANGE_SW[0], const.SF_small_RANGE_NE[0]])
    #ax1.set_ylim([const.SF_small_RANGE_SW[1], const.SF_small_RANGE_NE[1]])
    #ax2.set_xlim([const.SF_small_RANGE_SW[0], const.SF_small_RANGE_NE[0]])
    #ax2.set_ylim([const.SF_small_RANGE_SW[1], const.SF_small_RANGE_NE[1]])

    #ax1.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    #ax1.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])
    ax2.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax2.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])


    plt.show()

if __name__ == "__main__":
    sys.exit(main())
