#!/usr/bin/env python
"""
Created on Mon Oct 29, 2013

@author: ChenChen
"""

import sys
import cPickle
import copy

import matplotlib.pyplot as plt
from matplotlib import cm as CM
import numpy as np

import gps_track
import const

ANGLE_THRESHOLD = np.pi / 2.0

class LinerCurveModel:
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
        ones = np.ones(data.shape[0])
        A = np.vstack((data[:,0], data[:,1], ones)).T
        u, s, v = np.linalg.svd(A)
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
        e_min = min(data[:,0])
        e_max = max(data[:,0])
        n_min = min(data[:,1])
        n_max = max(data[:,1])

        if abs(model[0]) < abs(model[1]*10):
            px = np.linspace(e_min, e_max, num=100, endpoint = True)
            py = -1*model[2] - model[0]*px
            py /= model[1]
        else:
            py = np.linspace(n_min, n_max, num=100, endpoint=True)
            px = -1*model[2] - model[1]*py
            px /= model[0]

        return px, py

def ransac_line_fitting(data, init_guess, k, t, d):
    """ RANSAC Line fitting with initial guess
    """
    linear_model = LinerCurveModel()

    dist_to_line = np.abs(np.dot(data[:,0:2], np.array([init_guess[0], init_guess[1]])) + init_guess[2])
    all_idxs = np.arange(data.shape[0])

    line_dir = np.array([-1*init_guess[1], init_guess[0]])
    
    closeby_data_idxs = []
    for idx in all_idxs:
        if dist_to_line[idx] < t:
            data_point_dir = data[idx, 2:4]
            delta_angle = np.arccos(np.dot(data_point_dir, line_dir))
            if delta_angle < ANGLE_THRESHOLD:
                closeby_data_idxs.append(idx)

    closeby_data_idxs = np.array(closeby_data_idxs)
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
    dist_to_line = np.dot(data[:,0:2], np.array([line_param[0], line_param[1]])) + line_param[2]
    dist_to_line = np.abs(dist_to_line)
    dist_to_line /= np.linalg.norm([line_param[0], line_param[1]])

    line_norm = np.array([line_param[0], line_param[1]])
    line_norm /= np.linalg.norm(line_norm)
    line_dir = np.array([-1*line_norm[1], line_norm[0]])

    all_idxs = np.arange(data.shape[0])
    closeby_data_idxs = []
    for idx in all_idxs:
        if dist_to_line[idx] < t:
            data_point_dir = data[idx, 2:4]
            delta_angle = np.arccos(np.dot(data_point_dir, line_dir))
            if delta_angle < ANGLE_THRESHOLD:
                closeby_data_idxs.append(idx)
    closeby_data_idxs = np.array(closeby_data_idxs)
    
    if len(closeby_data_idxs) < 10:
        if with_idx:
            return False, [], []
        else:
            return False, []

    closeby_data = data[closeby_data_idxs, :]
    
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

def hough_line_fitting_with_direction(data, 
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
    
    if data.shape[1] < 4:
        print "Error! The data is in wrong format!"
        sys.exit()

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
            selected_angle_idx = []
            if np.linalg.norm(data[picked_idx,2:4]) < 1e-5:
                selected_angle_idx = range(0, n_theta_bin)
            else:
                dir_angle = np.arccos(data[picked_idx,2])
                if data[picked_idx,3] < 0:
                    dir_angle = 2*np.pi - dir_angle
                for idx in range(0, n_theta_bin):
                    if abs(theta[idx] - dir_angle) <= ANGLE_THRESHOLD\
                       or abs(theta[idx] - dir_angle + 2*np.pi) <= ANGLE_THRESHOLD\
                       or abs(dir_angle - theta[idx] + 2*np.pi) <= ANGLE_THRESHOLD:
                        selected_angle_idx.append(idx)
            for idx in selected_angle_idx:
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

        if len(line_param) == 0:
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

def main():
    with open(sys.argv[1], "r") as fin:
        data = cPickle.load(fin)
  
    tmp_data = copy.deepcopy(data)
    
    #hough_line_fitting(data, n_theta_bin, delta_dist, n_min, min_seg_length,
    #                   k, t, d):
    segments = hough_line_fitting_with_direction(tmp_data, 100, 25, 100, 250,
                       100, 35, 10)

    print "Totally %d line segments extracted."%len(segments)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(data[:,0], data[:,1], '.', color='gray')
    for seg in segments:
        ax.plot([seg[0], seg[2]],
                [seg[1], seg[3]],
                'x-', linewidth=3)

    plt.show()

if __name__ == "__main__":
    sys.exit(main())

