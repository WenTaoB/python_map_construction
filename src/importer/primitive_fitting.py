#!/usr/bin/env python
"""
Created on Mon Oct 07 17:19:49 2013

@author: ChenChen
"""

import sys
import math
import random
import cPickle
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from shapely.geometry import LineString
from scipy import spatial

import const
import gps_track

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
        
        err_per_point = sum(error) / len(error)

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
            print "hello"
            py = np.linspace(n_min, n_max, num=100, endpoint=True)
            px = -1*model[2] - model[1]*py
            px /= model[0]

        return px, py

    def get_segment(self, data, model):
        line_norm = np.array([model[0], model[1]])
        line_norm /= np.linalg.norm(line_norm)
        line_dir = np.array([-1*line_norm[1], line_norm[0]])

        k = -1*model[2] / (model[0]*line_norm[0] + model[1]*line_norm[1])
        xc = k*line_norm
        modified_x = data[:,0] - xc[0]
        modified_y = data[:,1] - xc[1]
        modified_data = np.array([modified_x, modified_y]).T

        tmp_data_projection = np.dot(modified_data, line_dir)
        
        sorted_idx = np.argsort(tmp_data_projection)
        data_projection = tmp_data_projection[sorted_idx]
        
        start_loc = data_projection[0]*line_dir + xc
        end_loc = data_projection[-1]*line_dir + xc

        return [start_loc, end_loc]

def extract_GPS_points_in_region(tracks, center, box_size):
    """
        Extract GPS points from a test region.
        Args:
            - tracks: GPS tracks;
            - center: a tuple, (center_x, center_y);
            - box_size: the length of the edge of the bounding box, in meters.
        Return:
            - a list of GPS points:
                [(p1_e, p1_n), (p2_e, p2_n), ...]
    """

    e_min = center[0] - box_size*0.5
    e_max = center[0] + box_size*0.5
    n_min = center[1] - box_size*0.5
    n_max = center[1] + box_size*0.5

    point_collection = []
    for track in tracks:
        for pt in track.utm:
            if pt[0] <= e_max and pt[0] >= e_min and\
                    pt[1] <= n_max and pt[1] >= n_min:
                point_collection.append(pt)

    return point_collection

def primitive_fitting(data,
                      n_pt_to_start,
                      dist_threshold,
                      min_accepting_rate,
                      max_n_iter,
                      fitting_model):
    """
        Fit primitives to data.
            Args:
                - data: numpy array (m, 2), where m is the number of GPS points;
                - dist_threshold: distance threshold in meters, distance larger than this will
                                  be neglected by the model;
                - min_accepting_rate: e.g., 0.5, a threshold, if the percentage of the assigned
                                      points exceed this value, the algorithm will exit.
            Return:
                - primitives: a list containing all primitives.
    """

    MAX_PRIMITIVE = 4
    
    for n_primitive in range(MAX_PRIMITIVE, MAX_PRIMITIVE+1):
        tmp_data = copy.deepcopy(data)
        models = []
        total_covered_pt = 0
        for model_i in np.arange(n_primitive):
            model, data_idxs, covered_pt = ransac_primitive_fitting(tmp_data, 
                                                             n_pt_to_start,
                                                             dist_threshold,
                                                             max_n_iter,
                                                             n_pt_to_start,
                                                             fitting_model)

            models.append(model)

            # Remove covered data point
            test_err = fitting_model.get_error(tmp_data, model)
            all_idxs = np.arange(tmp_data.shape[0])
            remaining_idxs = all_idxs[test_err >= dist_threshold]
            new_tmp_data = copy.deepcopy(tmp_data[remaining_idxs, :])
            tmp_data = new_tmp_data

            total_covered_pt += len(data_idxs)

        covered_ratio = float(total_covered_pt) / float(len(data))

        print "Using %d primitives:"%n_primitive
        print "\tCovered ratio = %.2f"%covered_ratio

        if covered_ratio >= min_accepting_rate:
            return models

    return models 

def ransac_primitive_fitting(data, 
                             n_pt_to_start,
                             dist_threshold,
                             n_iter,
                             min_n,
                             model):
    """
        Using RANSAC framework to fit n_primitive according to the model.
            Args:
                - data: numpy array of data (m, 2);
                - n_pt_to_start: the minimum number of data values required to fit the model;
                - dist_threshold: distance threshold in meters, distance larger than this will
                                  be neglected by the model;
                - n_iter: number of iterations to take;
                - model: model class, for example LinerCurveModel.
            Return:
                - model: a list of n_primitive models;
                - model_data_idxs: a list of data point belongs to the corresponding
                                    models;
                - total_error: total cumulative errors (in meters).
    """
    best_error = np.inf
    
    i_iter = 0

    print "ransac, shape = ", data.shape
    while i_iter < n_iter:
        this_error = 0.0

        maybe_idxs, test_idxs = random_partition(n_pt_to_start, 
                                                 data.shape[0])

        fitting_data = data[maybe_idxs, :]
        maybe_model = model.fit(fitting_data)

        test_data = data[test_idxs,:]
        test_err = model.get_error(test_data, maybe_model)
        also_idxs = test_idxs[test_err < dist_threshold]
        also_inliers = data[also_idxs, :]

        better_data = np.concatenate((fitting_data, also_inliers))

        better_model = model.fit(better_data)

        test_err = model.get_error(better_data, better_model)
        this_error = sum(test_err)

        test_err = model.get_error(data, better_model)
        all_idxs = np.arange(data.shape[0])
        inlier_idxs = all_idxs[test_err < dist_threshold]
        this_covered = len(inlier_idxs)

        this_error /= float(this_covered)

        if better_data.shape[0] >= 2*n_pt_to_start:
            if this_error < best_error: 
                best_model = better_model 
                best_model_data_idxs = inlier_idxs
                best_error = this_error

        i_iter += 1
    
    return best_model, best_model_data_idxs, best_error 

def random_partition(n_point, 
                     n_data):
    """
        Randomly partition the data for model fitting
            Args:
                - n_point: the number of data point assigned to each model;
                - n_data: the total number of data points.
            Return:
                - idxs: a list, each element is itself a index list, i.e., the data point idxs
                        assigned to the corresponding model;
                - test_idxs: data points left for testing.
    """
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n_point]
    idxs2 = all_idxs[n_point:]
    return idxs1, idxs2

def nearby_points_to_track(track, point_collection, search_range):
    """ Search nearby points to a track with a searching range.
        
        Args:
            - track: a single track of class GpsTrack;
            - point_collection: ndarray, of shape (m,2), i.e., the GPS location of data points;
            - search_range: in meters.
        Return:
            - nearby_points: ndarray, of shape (n,2).

    """

    kdtree = spatial.cKDTree(point_collection)

    data_points = []
    for pt in track.utm:
        data_points.append((pt[0], pt[1]))

    linestring = LineString(data_points)
    linestring.simplify(search_range)
    
    easting,northing = linestring.xy

    # Linear interpolation of the line string
    new_easting = []
    new_northing = []
    for i in np.arange(len(easting)-1):
        new_easting.append(easting[i])
        new_northing.append(northing[i])
        vec = np.array([easting[i+1] - easting[i], northing[i+1] - northing[i]])
        n_pt_to_insert = int(np.linalg.norm(vec)*2/search_range)
        orig = np.array([easting[i], northing[i]])
        for j in np.arange(n_pt_to_insert):
            t = 1.0 / float(n_pt_to_insert+1)
            pt = orig + t*(j+1)*vec
            new_easting.append(pt[0])
            new_northing.append(pt[1])
    new_easting.append(easting[-1])
    new_northing.append(northing[-1])

    nearby_idxs = {}

    for i in np.arange(len(new_easting)):
        idxs = kdtree.query_ball_point([new_easting[i], new_northing[i]], 
                                                 search_range)
        for idx in idxs:
            nearby_idxs[idx] = 1
    
    nearby_points = point_collection[nearby_idxs.keys(),:]
    print nearby_points.shape
    return nearby_points

def main():
    #tracks = gps_track.load_tracks(sys.argv[1])

    #CENTER_PT = (447820, 4423040)
    #BOX_SIZE = 1000

    #point_collection = extract_GPS_points_in_region(tracks, CENTER_PT, BOX_SIZE)

    #with open("test_region_points.dat", "w") as fout:
    #    cPickle.dump(point_collection, fout, protocol=2)

    #return

    #with open("test_region_points.dat", "r") as fin:
    #    point_collection = cPickle.load(fin)

    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    #ax.plot([pt[0] for pt in point_collection],
    #        [pt[1] for pt in point_collection],
    #        '.k')
    #plt.show()
    #
    #return
    #tracks = gps_track.load_tracks("test_region_tracks_clean.dat")
    #with open("test_region_points_array.dat", "r") as fin:
    #    point_collection = cPickle.load(fin)

    #track_idx = 13
    #search_range = 50

    #nearby_points = nearby_points_to_track(tracks[track_idx], 
    #                                       point_collection,
    #                                       search_range)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    #ax.plot(point_collection[:,0], point_collection[:,1], '.', color='gray')
    #ax.plot([pt[0] for pt in tracks[track_idx].utm],
    #        [pt[1] for pt in tracks[track_idx].utm],
    #        '.-r')
    #ax.plot(nearby_points[:,0],
    #        nearby_points[:,1],
    #        '.b')

    #plt.show()
    #return

    with open("test_region_points.dat", "r") as fin:
        point_collection = cPickle.load(fin)

    data = np.array([[pt[0] for pt in point_collection], [pt[1] for pt in point_collection]]).T

    model = LinerCurveModel()

    max_n_iter = 100
    n_pt_to_start = 100

    dist_threshold = 25
    min_accepting_rate = 0.8

    fitted_models = primitive_fitting(data,
                                      n_pt_to_start,
                                      dist_threshold,
                                      min_accepting_rate,
                                      max_n_iter,
                                      model)

    model_data_idxs = []
    for fitted_model in fitted_models:
        test_err = model.get_error(data, fitted_model)
        all_idxs = np.arange(data.shape[0])
        data_idx = all_idxs[test_err < dist_threshold]
        model_data_idxs.append(data_idx)

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(data[:,0], data[:,1], '.', color='gray')
    for i in range(0, len(fitted_models)):
        px, py = model.visualize(data[model_data_idxs[i],:], fitted_models[i])
        ax.plot(data[model_data_idxs[i],0], data[model_data_idxs[i], 1], '.r')
        ax.plot(px, py, '-')
        
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
