#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import sys
import cPickle
import random
import time
import copy
from optparse import OptionParser

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.collections import LineCollection
from scipy import spatial

from skimage.transform import hough_line,hough_line_peaks, probabilistic_hough_line
from skimage.filter import gaussian_filter, denoise_bilateral
import skimage.morphology as morphology

import gps_track
from point_cloud import PointCloud

import const

def grid_img(point_cloud,
             grid_size,
             loc,
             R,
             threshold,
             sigma=5):
    """ Sample the input point cloud using a uniform grid. 
        Args:
            - point_cloud: an object of PointCloud class
            - grid_size: in meters
            - loc: center
            - R: diameter
            - sigma: gaussian distribution variance, in meters
            - threshold: minimum value to count the box as one
        Return:
            - results: ndarray, rectangular image.
    """
    sample_points = []
    sample_directions = []
    
    min_easting = loc[0]-R
    max_easting = loc[0]+R
    min_northing = loc[1]-R
    max_northing = loc[1]+R

    n_grid_x = int((max_easting - min_easting)/grid_size + 0.5)
    n_grid_y = int((max_northing - min_northing)/grid_size + 0.5)

    print "Will generate image of size (%d, %d)"%(n_grid_x, n_grid_y)

    results = np.zeros((n_grid_x+1, n_grid_y+1))

    if n_grid_x > 1E4 or n_grid_y > 1E4:
        print "ERROR! The sampling grid is too small!"
        sys.exit(1)
   
    three_sigma = 3*sigma/grid_size
    
    geo_hash = {}
    for pt_idx in range(0, len(point_cloud.locations)):
        pt = point_cloud.locations[pt_idx]

        px = int((pt[0] - min_easting) / grid_size)
        py = int((pt[1] - min_northing) / grid_size)

        if px<0 or px>=n_grid_x or py<0 or py>=n_grid_y:
            continue

        # Expand around neighbor 
        pt_dir = point_cloud.directions[pt_idx]
        if np.linalg.norm(pt_dir) > 0.1:
            delta_x = np.dot(three_sigma*pt_dir, np.array([1.0, 0.0]))
            delta_y = np.sqrt(three_sigma**2 - delta_x**2)
            larger_one = max(abs(delta_x), abs(delta_y))
            n_pt_to_add = int(larger_one*2 + 1.5)

            tmp_i = np.linspace(px-delta_x, px+delta_x, n_pt_to_add)
            tmp_j = np.linspace(py-delta_y, py+delta_y, n_pt_to_add)

            for s in range(0, n_pt_to_add):
                i = int(tmp_i[s])
                j = int(tmp_j[s])

                if i<0 or i>=n_grid_x or j<0 or j>n_grid_y:
                    continue
                if geo_hash.has_key((i,j)):
                            geo_hash[(i,j)] += 1.0
                else:
                    geo_hash[(i,j)] = 1.0
        else:
            if geo_hash.has_key((px,py)):
                geo_hash[(px,py)] += 1.0
            else:
                geo_hash[(px,py)] = 1.0

    for key in geo_hash.keys():
        if geo_hash[key] >= threshold:
            results[key[0], key[1]] = geo_hash[key]

    filtered_img = results>0.9
    
    filtered_img = morphology.dilation(filtered_img, morphology.square(3))
    filtered_img = morphology.erosion(filtered_img, morphology.square(3))
    filtered_img = morphology.remove_small_objects(filtered_img, 10)

    results = filtered_img>0.9

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

def extract_line_segments(image, 
                          grid_size, 
                          loc, 
                          R, 
                          line_gap, 
                          search_range, 
                          p_removal,
                          display=False):
    """
        Extract line segments from an image file
            Args:
                - image: ndarray image data
                - grid_size: size of each pixel
                - loc: center of the actual region
                - R: radius of the actual region
                - line_gap: maximum gap in meters
                - search_range: used in pixel removal
                - p_removal: probability to remove a pixel
    """
    line_gap_in_pixel = int(line_gap/grid_size+0.5)

    all_lines = []
    lines = probabilistic_hough_line(image, 
                                     line_length=100,
                                     line_gap=line_gap_in_pixel)
   
    if display:
        fig = plt.figure(figsize=const.figsize)
        ax = fig.add_subplot(111)
        ax.imshow(image.T, cmap='gray')
        for line in lines:
            ax.plot([line[0][1], line[1][1]],
                    [line[0][0], line[1][0]], 'r-', linewidth=2)

        ax.set_xlim([0, image.shape[0]])
        ax.set_ylim([0, image.shape[1]])
        plt.show()

    all_lines.extend(lines)
    modified_img1 = remove_pixels(image, 
                                  lines, 
                                  p_removal=p_removal,
                                  search_range=search_range)
    modified_img1 = morphology.remove_small_objects(modified_img1, 10)

    new_lines1 = probabilistic_hough_line(modified_img1, 
                                          line_length=50,
                                          line_gap=line_gap_in_pixel)

    if display:
        fig = plt.figure(figsize=const.figsize)
        ax = fig.add_subplot(111)
        ax.imshow(modified_img1.T, cmap='gray')
        for line in new_lines1:
            ax.plot([line[0][1], line[1][1]],
                    [line[0][0], line[1][0]], 'r-', linewidth=2)

        ax.set_xlim([0, image.shape[0]])
        ax.set_ylim([0, image.shape[1]])
        plt.show()
 
    all_lines.extend(new_lines1)
    modified_img2 = remove_pixels(modified_img1,
                                  new_lines1,
                                  p_removal=p_removal,
                                  search_range=search_range)
    modified_img2 = morphology.remove_small_objects(modified_img2, 20)

    new_lines2 = probabilistic_hough_line(modified_img2,
                                          line_length=20,
                                          line_gap=line_gap_in_pixel)
    all_lines.extend(new_lines2)

    return all_lines

    if display:
        fig = plt.figure(figsize=const.figsize)
        ax = fig.add_subplot(111)
        ax.imshow(modified_img2.T, cmap='gray')
        for line in new_lines2:
            ax.plot([line[0][1], line[1][1]],
                    [line[0][0], line[1][0]], 'r-', linewidth=2)

        ax.set_xlim([0, image.shape[0]])
        ax.set_ylim([0, image.shape[1]])
        plt.show()
 
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
                                  grid_size,
                                  loc,
                                  R,
                                  threshold = 1):
    """ Filter GPS point_cloud using a grid.
        Args:
            - point_cloud: GPS point cloud
            - grid_size: grid size in meters
            - loc: center of the region
            - R: radius of the region
        Return:
            - sample_point_cloud: an object of PointCloud.
    """
    min_easting = loc[0]-R
    max_easting = loc[0]+R
    min_northing = loc[1]-R
    max_northing = loc[1]+R

    n_grid_x = int((max_easting - min_easting)/grid_size + 0.5)
    n_grid_y = int((max_northing - min_northing)/grid_size + 0.5)
    
    if n_grid_x > 1E4 or n_grid_y > 1E4:
        print "ERROR in filter_point_cloud_using_grid! The sampling grid is too small!"
        sys.exit(1)
    
    geo_hash = {}
    dir_hash = {}
    geo_hash_count = {} 
    geo_hash_direction = {}
    # Traversing through all GPS points
    for pt_idx in range(0, len(point_cloud.locations)):
        pt = point_cloud.locations[pt_idx]
        pt_dir = point_cloud.directions[pt_idx]
        pt_dir_norm = np.linalg.norm(pt_dir)

        px = int((pt[0] - min_easting) / grid_size)
        py = int((pt[1] - min_northing) / grid_size)

        if px<0 or px>n_grid_x or py<0 or py>n_grid_y:
            print "ERROR! Point outside the grid!"
            sys.exit(1)

        if geo_hash.has_key((px, py)):
            geo_hash_count[(px, py)] += 1
            geo_hash[(px, py)] += pt
            if pt_dir_norm > 0.1:
                geo_hash_direction[(px, py)].append(np.copy(pt_dir))
        else:
            geo_hash_count[(px, py)] = 1.0
            geo_hash[(px, py)] = np.copy(pt)
            if pt_dir_norm > 0.1:
                geo_hash_direction[(px, py)] = [pt_dir]
            else:
                geo_hash_direction[(px, py)] = []
   
    # Deal with each non-empty box
    sample_point_locations = []
    sample_point_directions = []
    for key in geo_hash.keys():
        if geo_hash_count[key] < threshold:
            continue

        pt = geo_hash[key] / geo_hash_count[key]
        directions = geo_hash_direction[key]

        if len(directions) == 0:
            sample_point_locations.append(pt)
            sample_point_directions.append(np.array([0.0, 0.0]))
            continue

        # Clustering the directions
        n_angle_bin = 16
        delta_angle = 2*np.pi/n_angle_bin
        angle_directions = []
        tmp_directions = list(directions)
        while True:
            angle_bin_count = np.zeros(n_angle_bin)
            for direction in tmp_directions:
                angle = np.arccos(np.dot(direction, np.array([1.0, 0.0])))
                if direction[1] < 0:
                    angle = 2*np.pi - angle
                angle_bin_idx = int(angle / delta_angle)
                angle_bin_count[angle_bin_idx] += 1
            # Find max
            max_idx = np.argmax(angle_bin_count)
            
            if angle_bin_count[max_idx] < 1:
                break

            max_angle_dir = (max_idx+0.5)*delta_angle
            max_angle_vec = np.array([np.cos(max_angle_dir), np.sin(max_angle_dir)])
            
            new_directions = []
            result_direction = np.array([0.0, 0.0])
            for direction in tmp_directions:
                if np.dot(direction, max_angle_vec) > np.cos(np.pi/12.0):
                    result_direction += direction
                else:
                    new_directions.append(np.copy(direction))
            tmp_directions = new_directions
            result_direction /= np.linalg.norm(result_direction)
            sample_point_locations.append(np.copy(pt))
            sample_point_directions.append(np.copy(result_direction))

    sample_point_cloud = PointCloud(np.array(sample_point_locations),
                                    np.array(sample_point_directions))
    return sample_point_cloud

def correct_direction_with_lines(sample_point_cloud,
                                 lines,
                                 search_radius=15,
                                 angle_threshold=np.pi/6.0):
    """ Correct sample_point_cloud using the extracted lines
        Args:
            - sample_point_cloud: an object of PointCloud
            - lines: an array of [(p_start_e, p_start_n), (p_end_e, p_end_n)]
            - search_radius: distance to search, in meters. Default is 10m.
        Return:
            - new_sample_point_cloud: an object of PointCloud
    """
    new_sample_locations = [] 
    new_sample_directions = []
    
    line_starts = lines[:,0,:]
    line_ends = lines[:,1,:]
    line_vecs = line_ends - line_starts
    
    sample_point_hash = {}
    for pt_idx in range(0, sample_point_cloud.locations.shape[0]):
        if np.linalg.norm(sample_point_cloud.locations[pt_idx]) < 0.1:
            new_sample_locations.append(sample_point_cloud.locations[pt_idx])
            new_sample_locations.append(sample_point_cloud.directions[pt_idx])

        pt = sample_point_cloud.locations[pt_idx]
        pt_key = (int(pt[0]), int(pt[1]))
        # Find nearby line segments
        nearby_line_idxs = []
        vec1s = pt - line_starts
        vec2s = pt - line_ends

        for line_idx in range(0, len(vec1s)):
            if np.dot(vec1s[line_idx], vec2s[line_idx]) > 0:
                continue

            line_dir = line_vecs[line_idx] / np.linalg.norm(line_vecs[line_idx])
            line_norm = np.array([-1*line_dir[1], line_dir[0]])
            dist = abs(np.dot(vec1s[line_idx], line_norm))
            if dist <= search_radius:
                nearby_line_idxs.append(line_idx)

        if len(nearby_line_idxs) == 0:
            continue

        # Check if direction is consistent
        potential_direction = np.array([0.0, 0.0])
        direction_acceptable = False
        for line_idx in nearby_line_idxs:
            line_dir = line_vecs[line_idx] / np.linalg.norm(line_vecs[line_idx])
            dot_value = np.dot(line_dir, sample_point_cloud.directions[pt_idx])
            if abs(dot_value) > np.cos(angle_threshold):
                direction_acceptable = True
                if dot_value > 0:
                    potential_direction += line_dir
                else:
                    potential_direction += -1*line_dir
        
        if direction_acceptable:
            potential_direction /= np.linalg.norm(potential_direction)
            if not sample_point_hash.has_key(pt_key):
                sample_point_hash[pt_key] = [potential_direction]
                new_sample_locations.append(sample_point_cloud.locations[pt_idx])
                new_sample_directions.append(potential_direction)
            else:
                should_insert = True
                for pt_direction in sample_point_hash[pt_key]:
                    if np.dot(pt_direction, potential_direction) > np.cos(angle_threshold):
                        should_insert = False
                if should_insert:
                    new_sample_locations.append(sample_point_cloud.locations[pt_idx])
                    new_sample_directions.append(potential_direction)
                    sample_point_hash[pt_key].append(potential_direction)
                
    return PointCloud(np.array(new_sample_locations), np.array(new_sample_directions))

def visualize_sample_point_cloud(sample_point_cloud, 
                                 loc, 
                                 R):
    fig = plt.figure(figsize=const.figsize)
    
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot(sample_point_cloud.locations[:,0],
            sample_point_cloud.locations[:,1],
            '.', color='gray')
    arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False}
    for i in range(0, sample_point_cloud.locations.shape[0]):
        direction = sample_point_cloud.directions[i]
        if np.linalg.norm(direction) < 0.1:
            continue
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
    parser = OptionParser()
    parser.add_option("-p","--pointcloud", dest="point_cloud", help="Point cloud filename.", type="string", metavar="PointCloudFile")
    parser.add_option("-o", dest="output_img", help="Generated line feature img.")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)
    parser.add_option("--gridsize", dest="grid_size", type="float", default=2.5, help="Grid size in meters, default=2.5m", metavar="GRID_SIZE")

    (options, args) = parser.parse_args()

    if not options.point_cloud:
        parser.error("Point cloud filename not found!")
    if not options.output_img:
        parser.error("Output image filename not found!")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)
    
    GRID_SIZE = options.grid_size # in meters

    # Load point cloud
    with open(options.point_cloud, "rb") as fin:
        point_cloud = cPickle.load(fin)
    print "There are %d points in the point cloud."%point_cloud.locations.shape[0]

    # Get image
    THRESHOLD = 1.0
    img = grid_img(point_cloud,
                   GRID_SIZE,
                   LOC,
                   R,
                   THRESHOLD,
                   sigma=2.0)
    
    # Extract line segments from image
    LINE_GAP = 10
    SEARCH_RANGE = 5
    P_REMOVAL = 0.1   
    lines = extract_line_segments(img,
                                  GRID_SIZE,
                                  LOC,
                                  R,
                                  LINE_GAP,
                                  SEARCH_RANGE,
                                  P_REMOVAL)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.Axes(fig, [0., 0., 1., 1.], aspect='equal')
    ax.set_axis_off()
    fig.add_axes(ax)

    ROAD_WIDTH = 4 # in meters
    #ax.imshow(img.T, cmap='gray')
    for line in lines:
        ax.plot([line[0][1], line[1][1]],
                [line[0][0], line[1][0]], 'k-', linewidth=2)
    ax.set_xlim([0, img.shape[0]])
    ax.set_ylim([0, img.shape[1]])
    fig.savefig(options.output_img, dpi=10)
    plt.close()

    return
    
if __name__ == "__main__":
    sys.exit(main())
