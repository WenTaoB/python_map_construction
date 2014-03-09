#!/usr/bin/env python
"""
Using a unified grid to rasterize GPS tracks.
"""

import sys
import time
import cPickle
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from sklearn.decomposition import PCA

import networkx as nx

import gps_track
import const

class PointCloud:
    def __init__(self, locations, directions, track_ids):
        self.locations = np.array(locations)
        self.directions = np.array(directions)
        self.track_ids = np.array(track_ids)

def extract_point_cloud(tracks, loc, R):
    locations = []
    directions = []
    track_ids = []
    for track_idx in range(0, len(tracks)):
        track = tracks[track_idx]
        for pt_idx in range(0, len(track.utm)):
            pt = track.utm[pt_idx]
            if pt[0]>=loc[0]-R and pt[0]<=loc[0]+R and \
                    pt[1]>=loc[1]-R and pt[1]<=loc[1]+R:
                locations.append((pt[0], pt[1]))
               
                dir1 = np.array((0.0, 0.0))
                if pt_idx > 0:
                    dir1 = np.array((track.utm[pt_idx][0]-track.utm[pt_idx-1][0], track.utm[pt_idx][1]-track.utm[pt_idx-1][1]))

                dir2 = np.array((0.0, 0.0)) 
                if pt_idx < len(track.utm) - 1:
                    dir2 = np.array((track.utm[pt_idx+1][0]-track.utm[pt_idx][0], track.utm[pt_idx+1][1]-track.utm[pt_idx][1]))

                direction = dir1 + dir2
                
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1.0:
                    direction /= direction_norm
                else:
                    direction *= 0.0
                
                directions.append(direction)
                track_ids.append(track_idx)
    
    point_cloud = PointCloud(locations, directions, track_ids)
    with open("test_point_cloud.dat", "wb") as fout:
        cPickle.dump(point_cloud, fout, protocol=2)

def filter_point_cloud_using_grid(point_cloud, sample_grid_size):
    """ Sample the input point cloud using a uniform grid. If there are points in the cell,
        we will use the average.
    """
    sample_points = []
    sample_directions = []
    min_easting = min(point_cloud.locations[:,0])
    max_easting = max(point_cloud.locations[:,0])
    min_northing = min(point_cloud.locations[:,1])
    max_northing = max(point_cloud.locations[:,1])

    n_grid_x = int((max_easting - min_easting)/sample_grid_size + 0.5)
    n_grid_y = int((max_northing - min_northing)/sample_grid_size + 0.5)

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
        pt = geo_hash[key] / geo_hash_count[key]
        sample_points.append(pt)
        dir_norm = np.linalg.norm(dir_hash[key])
        if dir_norm > 1.0:
            sample_directions.append(dir_hash[key]/dir_norm)
        else:
            sample_directions.append(np.array((0.0, 0.0)))

    sample_point_cloud = PointCloud(sample_points, sample_directions, [-1]*len(sample_points))
    return sample_point_cloud

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
    
def visualize_point_cloud(point_cloud, loc, R):
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(121, aspect="equal")
    ax.plot(point_cloud.locations[:,0], point_cloud.locations[:,1], '.')
    ax.set_xlim([loc[0]-R, loc[0]+R]) 
    ax.set_ylim([loc[1]-R, loc[1]+R]) 

    ax = fig.add_subplot(122, aspect="equal")
    ax.quiver(point_cloud.locations[:,0],
              point_cloud.locations[:,1],
              point_cloud.directions[:,0],
              point_cloud.directions[:,1]) 
    ax.set_xlim([loc[0]-R, loc[0]+R]) 
    ax.set_ylim([loc[1]-R, loc[1]+R]) 
    plt.show()

def sample_point_neighborhood_pca(point_cloud, 
                                  sample_point_cloud, 
                                  kdtree_point_cloud,
                                  R):
    kdtree_sample_point = spatial.cKDTree(sample_point_cloud.locations, leafsize=1) 
    principal_directions = []
    ratio = []
    for pt in sample_point_cloud.locations:
        neighbor_idx = kdtree_sample_point.query_ball_point(pt, R)
        neighbor_pts = np.copy(sample_point_cloud.locations[neighbor_idx, :])

        sum_vec = np.array((0.0, 0.0))
        for neighbor_pt in neighbor_pts:
            sum_vec += neighbor_pt
        mean_vec = sum_vec / len(neighbor_pts)
        
        for neighbor_pt in neighbor_pts:
            neighbor_pt -= mean_vec
        
        M = np.dot(neighbor_pts.T, neighbor_pts) / neighbor_pts.shape[0]
        u,s,v = np.linalg.svd(M)

        orientation = u[0,:]
        principal_directions.append(orientation)
        ratio.append(s[0]/s[1])

    return np.array(principal_directions)

    ratio = np.array(np.log(ratio))
    print "max = ", max(ratio)
    print "min = ", min(ratio)
    normalized_ratio = (ratio - min(ratio))/(max(ratio)-min(ratio))

    principal_directions = np.array(principal_directions)
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot(sample_point_cloud.locations[:,0], sample_point_cloud.locations[:,1], '.')
    ax.quiver(sample_point_cloud.locations[:,0],
              sample_point_cloud.locations[:,1],
              sample_point_cloud.directions[:,0],
              sample_point_cloud.directions[:,1]) 
    ax.quiver(sample_point_cloud.locations[:,0],
              sample_point_cloud.locations[:,1],
              principal_directions[:,0],
              principal_directions[:,1],
              color='r')
    loc = (447772, 4424300)
    # Target region radius
    R = 300
    ax.set_xlim([loc[0]-R, loc[0]+R]) 
    ax.set_ylim([loc[1]-R, loc[1]+R]) 
    plt.show()

    plt.show()

def grow_segment(starting_pt_idx,
                 sample_point_cloud, 
                 sample_point_kdtree,
                 principal_directions,
                 search_radius,
                 width_threshold,
                 angle_threshold,
                 min_pt_to_record):
    """ Grow segment from a starting point.
        Args:
            - starting_pt_idx: index in sample_point_cloud.locations
            - sample_point_kdtree: kdtree built from sample_point_cloud.locations
            - principal_directions: principal_directions for each sample points
            - search_radius: search radius each time to grow the segment
            - width_threshold: distance threshold to the line defined by the searching point
                               and its principal direction
            - angle_threshold: in radius, largest angle tolerance 
        Return:
            - result_segment_pt_idxs: a list of indices recording the sample points in the resulting segment
    """
    result_segment_pt_idxs = []
    segment_point_idxs_dict = {}
    front_dir = np.copy(principal_directions[starting_pt_idx])
    end_dir = -1*np.copy(principal_directions[starting_pt_idx])
    front_pt_idx = starting_pt_idx 
    end_pt_idx = starting_pt_idx
    front_stopped = False
    end_stopped = False
    while True:
        if not front_stopped:
            # Trace forward
            candidate_nearby_point_idxs = \
                sample_point_kdtree.query_ball_point(sample_point_cloud.locations[front_pt_idx],
                                                     search_radius)
            norm_dir = np.array([-1*front_dir[1], front_dir[0]])
            n_pt_to_add = 0

            nxt_front_pt_idx = -1
            nxt_front_pt_proj = 0.0
            for candidate_idx in candidate_nearby_point_idxs:
                if np.dot(sample_point_cloud.directions[candidate_idx], front_dir) < np.cos(angle_threshold):
                    continue
                if abs(np.dot(principal_directions[candidate_idx], front_dir)) > np.cos(angle_threshold):
                    vec = sample_point_cloud.locations[candidate_idx] - sample_point_cloud.locations[front_pt_idx]
                    pt_proj = np.dot(vec, front_dir)
                    if pt_proj > 0 and abs(np.dot(norm_dir, vec)) <= width_threshold:
                        segment_point_idxs_dict[candidate_idx] = 1
                        if pt_proj > nxt_front_pt_proj:
                            front_pt_proj = pt_proj
                            nxt_front_pt_idx = candidate_idx
                        n_pt_to_add += 1
            front_pt_idx = nxt_front_pt_idx 
            if n_pt_to_add == 0:
                front_stopped = True

            tmp_front_dir = np.copy(principal_directions[front_pt_idx])
            if np.dot(tmp_front_dir, front_dir) > 0:
                front_dir = tmp_front_dir
            else:
                front_dir = -1*tmp_front_dir
        
        # Trace endbackward
        if not end_stopped:
            candidate_nearby_point_idxs = \
                sample_point_kdtree.query_ball_point(sample_point_cloud.locations[end_pt_idx],
                                                     search_radius)
            norm_dir = np.array([-1*end_dir[1], end_dir[0]])
            n_pt_to_add = 0
            nxt_end_pt_idx = -1
            nxt_end_pt_proj = 0.0
            for candidate_idx in candidate_nearby_point_idxs:
                if np.dot(-1*sample_point_cloud.directions[candidate_idx], end_dir) < np.cos(angle_threshold):
                    continue
                if abs(np.dot(principal_directions[candidate_idx], end_dir)) > np.cos(angle_threshold):
                    vec = sample_point_cloud.locations[candidate_idx] - sample_point_cloud.locations[end_pt_idx]
                    pt_proj = np.dot(vec, end_dir)
                    if pt_proj > 0 and abs(np.dot(norm_dir, vec)) <= width_threshold:
                        segment_point_idxs_dict[candidate_idx] = 1
                        if pt_proj > nxt_end_pt_proj:
                            nxt_end_pt_proj = pt_proj
                            nxt_end_pt_idx = candidate_idx
                        n_pt_to_add += 1
            end_pt_idx = nxt_end_pt_idx 
            if n_pt_to_add == 0:
                end_stopped = True

            tmp_end_dir = np.copy(principal_directions[end_pt_idx])
            if np.dot(tmp_end_dir, end_dir) > 0:
                end_dir = tmp_end_dir
            else:
                end_dir = -1*tmp_end_dir
        if front_stopped and end_stopped:
            break

    if len(segment_point_idxs_dict.keys()) >= min_pt_to_record:
        result_segment_pt_idxs.append(copy.copy(segment_point_idxs_dict.keys()))

    segment_point_idxs_dict = {}
    front_dir = -1*np.copy(principal_directions[starting_pt_idx])
    end_dir = np.copy(principal_directions[starting_pt_idx])
    front_pt_idx = starting_pt_idx 
    end_pt_idx = starting_pt_idx
    front_stopped = False
    end_stopped = False
    while True:
        if not front_stopped:
            # Trace forward
            candidate_nearby_point_idxs = \
                sample_point_kdtree.query_ball_point(sample_point_cloud.locations[front_pt_idx],
                                                     search_radius)
            norm_dir = np.array([-1*front_dir[1], front_dir[0]])
            n_pt_to_add = 0

            nxt_front_pt_idx = -1
            nxt_front_pt_proj = 0.0
            for candidate_idx in candidate_nearby_point_idxs:
                if np.dot(sample_point_cloud.directions[candidate_idx], front_dir) < np.cos(angle_threshold):
                    continue
                if abs(np.dot(principal_directions[candidate_idx], front_dir)) > np.cos(angle_threshold):
                    vec = sample_point_cloud.locations[candidate_idx] - sample_point_cloud.locations[front_pt_idx]
                    pt_proj = np.dot(vec, front_dir)
                    if pt_proj > 0 and abs(np.dot(norm_dir, vec)) <= width_threshold:
                        segment_point_idxs_dict[candidate_idx] = 1
                        if pt_proj > nxt_front_pt_proj:
                            front_pt_proj = pt_proj
                            nxt_front_pt_idx = candidate_idx
                        n_pt_to_add += 1
            front_pt_idx = nxt_front_pt_idx 
            if n_pt_to_add == 0:
                front_stopped = True

            tmp_front_dir = np.copy(principal_directions[front_pt_idx])
            if np.dot(tmp_front_dir, front_dir) > 0:
                front_dir = tmp_front_dir
            else:
                front_dir = -1*tmp_front_dir
        
        # Trace endbackward
        if not end_stopped:
            candidate_nearby_point_idxs = \
                sample_point_kdtree.query_ball_point(sample_point_cloud.locations[end_pt_idx],
                                                     search_radius)
            norm_dir = np.array([-1*end_dir[1], end_dir[0]])
            n_pt_to_add = 0
            nxt_end_pt_idx = -1
            nxt_end_pt_proj = 0.0
            for candidate_idx in candidate_nearby_point_idxs:
                if np.dot(-1*sample_point_cloud.directions[candidate_idx], end_dir) < np.cos(angle_threshold):
                    continue
                if abs(np.dot(principal_directions[candidate_idx], end_dir)) > np.cos(angle_threshold):
                    vec = sample_point_cloud.locations[candidate_idx] - sample_point_cloud.locations[end_pt_idx]
                    pt_proj = np.dot(vec, end_dir)
                    if pt_proj > 0 and abs(np.dot(norm_dir, vec)) <= width_threshold:
                        segment_point_idxs_dict[candidate_idx] = 1
                        if pt_proj > nxt_end_pt_proj:
                            nxt_end_pt_proj = pt_proj
                            nxt_end_pt_idx = candidate_idx
                        n_pt_to_add += 1
            end_pt_idx = nxt_end_pt_idx 
            if n_pt_to_add == 0:
                end_stopped = True

            tmp_end_dir = np.copy(principal_directions[end_pt_idx])
            if np.dot(tmp_end_dir, end_dir) > 0:
                end_dir = tmp_end_dir
            else:
                end_dir = -1*tmp_end_dir
        if front_stopped and end_stopped:
            break

    if len(segment_point_idxs_dict.keys()) > min_pt_to_record:
        result_segment_pt_idxs.append(segment_point_idxs_dict.keys())

    return result_segment_pt_idxs

def gradient(points, sample_points, point_kdtree, sample_kdtree, radius, K):
    """ Compute the gradient at each variable
    """
    results = []
    sum_val = 0.0
    for i in range(0, sample_points.shape[0]):
        # Find nearby points
        d, point_idxs = point_kdtree.query(sample_points[i], 10)
        # Find nearby sample points
        dist, nb_point_idxs = sample_kdtree.query(sample_points[i], K)
        
        nearby_sample_points = sample_points[nb_point_idxs[1:],:]
        M = np.dot(nearby_sample_points.T, nearby_sample_points)

        u,s,v = np.linalg.svd(M)
        sigma = s[0] / sum(s)
        
        g0 = 0.0
        g1 = 0.0
        for idx in point_idxs:
            vec = sample_points[i] - points[idx]
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-3:
                vec_norm = 1e-3
            sum_val += vec_norm
            g0 += ((sample_points[i,0] - points[idx,0]) / vec_norm)
            g1 += ((sample_points[i,1] - points[idx,1]) / vec_norm)

        for ind in range(1, len(nb_point_idxs)):
            if dist[ind] < 1e-3:
                dist[ind] = 1e-3
            sum_val += 1.0/dist[ind]
            idx = nb_point_idxs[ind]
            g0 -= ((sample_points[i,0] - sample_points[idx,0]) / dist[ind]**3 / sigma)
            g1 -= ((sample_points[i,1] - sample_points[idx,1]) / dist[ind]**3 / sigma)

        results.append([g0, g1])

    return np.array(results), sum_val

def l1_segment_fitting(points, initial_sampling_rate):
    """ L1 skeleton fitting for the points.
        Args:
            - points: ndarray points
    """
    K = 10
    radius = 0.5
    contraction_loop_num = 100
    
    point_kdtree = spatial.cKDTree(points, leafsize=1)
    n_init_samples = int(len(points) * initial_sampling_rate)
    if n_init_samples < 1:
        n_init_samples = 1
    
    point_idxs = np.arange(points.shape[0])

    np.random.shuffle(point_idxs)
    init_samples = [] 
    for i in range(0, n_init_samples):
        init_samples.append((points[point_idxs[i],0], points[point_idxs[i],1]))
    init_samples = np.array(init_samples)
    samples = np.copy(init_samples) 

    for loop in range(0, contraction_loop_num):
        print loop
        
        sample_kdtree = spatial.cKDTree(samples, leafsize=1)
        print 'data=',sample_kdtree.data
        new_samples = []
        for i in range(0, samples.shape[0]):
            dist, nb_point_idxs = sample_kdtree.query(samples[i], K)
            print nb_point_idxs
            for j in range(1, len(nb_point_idxs)):
                if nb_point_idxs[j] == np.nan:
                    break
            nearby_sample_points = samples[nb_point_idxs[1:j],:]
            M = np.dot(nearby_sample_points.T, nearby_sample_points)

            u,s,v = np.linalg.svd(M)
            curvature = s[0] / sum(s)

            sample_pt = samples[i]
            idxs = point_kdtree.query_ball_point(sample_pt, radius)

            contractions = np.array([0.0, 0.0])
            alpha_weight = 0.0
            for j in range(0, len(idxs)):
                point_original = points[idxs[j]]
                distance = np.linalg.norm(sample_pt - point_original)
                theta = np.exp(-4.0*distance*distance/radius/radius)
                distance = max(distance, 0.001*radius)
                alpha = theta / distance
                contractions += point_original*alpha
                alpha_weight += alpha

            repulsions = np.array([0.0, 0.0])
            beta_weight = 0.0
            idxs = sample_kdtree.query_ball_point(sample_pt, radius)
            for j in range(0, len(idxs)):
                if idxs[j] == i:
                    continue
                
                distance = np.linalg.norm(sample_pt - samples[idxs[j]])
                theta = np.exp(-4.0*distance*distance/radius/radius)
                distance = max(distance, 0.001*radius)
                beta = theta / distance / distance
                repulsion = sample_pt - samples[idxs[j]]
                repulsions += beta*repulsion
                beta_weight += beta
           
            mu = 0.35
            new_pt = contractions/alpha_weight + repulsions*(mu*curvature/beta_weight)
            idxs = point_kdtree.query_ball_point(new_pt, radius)
            if len(idxs) != 0:
                new_samples.append(new_pt)

        new_samples = np.array(new_samples)
        samples = new_samples

    return init_samples, samples

def main():
    if len(sys.argv) != 2:
            print "ERROR!!!"
            return

    x = np.random.rand(100)
    y = 1+x + x*x + 0.5*np.random.rand(100)
    points = np.array((x,y)).T

    init_samples, new_samples = l1_segment_fitting(points, 0.1)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.', color='gray')

    ax.plot(init_samples[:,0], init_samples[:,1], 'ob')
    ax.plot(new_samples[:,0], new_samples[:,1], '+r', markersize=12)
    plt.show()
    return
    #tracks = gps_track.load_tracks(sys.argv[1])
    
    with open("test_point_cloud.dat", "rb") as fin:
        point_cloud = cPickle.load(fin)

    # Target location
    LOC = (447772, 4424300)
    # Target region radius
    R = 300

    #extract_point_cloud(tracks, LOC, R) 
    #visualize_point_cloud(point_cloud, LOC, R)

    sample_point_cloud = filter_point_cloud_using_grid(point_cloud, 5.0)

    #visualize_sample_points(point_cloud, sample_point_cloud, LOC, R)

    kdtree_point_cloud = spatial.cKDTree(point_cloud.locations, leafsize=1)
    principal_directions = sample_point_neighborhood_pca(point_cloud, sample_point_cloud, kdtree_point_cloud, 100.0)
    sample_point_kdtree = spatial.cKDTree(sample_point_cloud.locations, leafsize=1)

    # Grow segment using principal directions
    SEARCH_RADIUS = 20
    WIDTH_THRESHOLD = 10
    ANGLE_THRESHOLD = np.pi / 3.0
    MIN_PT_TO_RECORD = 5

    starting_pt_idx = np.random.randint(0, len(sample_point_cloud.directions))
    start_time = time.time()
    result_segments = grow_segment(starting_pt_idx,
                                      sample_point_cloud,
                                      sample_point_kdtree,
                                      principal_directions,
                                      SEARCH_RADIUS,
                                      WIDTH_THRESHOLD,
                                      ANGLE_THRESHOLD,
                                      MIN_PT_TO_RECORD)

    print "It took %f seconds"%(time.time() - start_time)
    print "There are %d segments"%(len(result_segments))
    for seg in result_segments:
        print "There are %d points in this segment."%len(seg)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot(sample_point_cloud.locations[:,0], 
            sample_point_cloud.locations[:,1], '.', color='gray')
    ax.plot(sample_point_cloud.locations[starting_pt_idx,0],
            sample_point_cloud.locations[starting_pt_idx,1], 'or')
    ax.quiver(sample_point_cloud.locations[starting_pt_idx,0],
              sample_point_cloud.locations[starting_pt_idx,1],
              principal_directions[starting_pt_idx,0],
              principal_directions[starting_pt_idx,1],color='r')

    for i in range(0, len(result_segments)):
        segment_point_idxs = result_segments[i]
        ax.plot(sample_point_cloud.locations[segment_point_idxs,0],
                sample_point_cloud.locations[segment_point_idxs,1],
                '.', color=const.colors[i])

    plt.show()
    #visualize_sample_points(point_cloud, sample_points, LOC, R)


if __name__ == "__main__":
    sys.exit(main())

