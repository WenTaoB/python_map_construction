#!/usr/bin/env python
"""
    Functions that can compute descriptors in the neighborhood of a given location.
"""

import glob
import re
import sys
import cPickle
import math

import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib.collections import LineCollection
import numpy as np

import gps_track
import const

def neighborhood_point_density(points,
                               center,
                               sigma_neighborhood,
                               sigma_gps_point,
                               n_bin):
    """
        Compute 2d point density histogram.
        Assumptions:
            - Density is weighted by 2d Gaussian: N(center, sigma_neighborhood);
            - Density for each points is modeled as N(gps_point, sigma_gps_point);
            - Finally the matrix is normalized so that it sums up to 1.
        Params:
            - points: list of GPS points of form [(p1_easting, p1_northing), ...];
            - center: center location, e.g., (c_easting, c_northing);
            - sigma_neighborhood: in meters;
            - sigma_gps_point: in meters;
            - n_bin: the returning matrix will be of shape (2*n_bin+1, 2*n_bin+1).
        Return:
            - hist: a (2*n_bin+1, 2*n_bin+1) numpy matrix, with center point at its center.
    """
    delta_d = float(sigma_neighborhood) / n_bin
    total_bins = 2*n_bin + 1

    hist = np.zeros((total_bins, total_bins))

    bin_center_E = np.ones((total_bins,total_bins))
    bin_center_N = np.ones((total_bins,total_bins))

    bin_center_E[:,n_bin] *= center[0]
    bin_center_N[n_bin,:] *= center[1]

    for i in range(1, n_bin+1):
        bin_center_N[n_bin+i,:] *= (center[1]+delta_d*i)
        bin_center_N[n_bin-i,:] *= (center[1]-delta_d*i)

        bin_center_E[:,n_bin+i] *= (center[0]+delta_d*i)
        bin_center_E[:,n_bin-i] *= (center[0]-delta_d*i)

    for pt in points:
        influence = np.exp(-1*(bin_center_E-pt[0])**2/2.0/sigma_gps_point/sigma_gps_point)*\
                    np.exp(-1*(bin_center_N-pt[1])**2/2.0/sigma_gps_point/sigma_gps_point)
        hist += influence

    mask = np.exp(-1*(bin_center_E-center[0])**2/2.0/sigma_neighborhood/sigma_neighborhood)*\
           np.exp(-1*(bin_center_N-center[1])**2/2.0/sigma_neighborhood/sigma_neighborhood)
    hist *= mask
    hist /= sum(sum(hist))
    return np.log(hist+0.01)

def angle_histogram(tracks,
                    n_bin):
    """
        Compute angle histogram of a set of tracks. Angle is defined as follows:
            For p_i on track t,
                v_i = p_(i+1) - p_(i-1).
                If i = 0, or end of track, v_i is the vector between its neighboring point.
        Args:
            - tracks: a list of GPS tracks;
            - n_bin: number of bins between [0,2*pi]
        Return:
            - angles
            - result: normalized histogram vector.
    """
    results = np.zeros(n_bin)
    delta_angle = 2.0 * np.pi / n_bin
    angles = delta_angle / 2.0 / np.pi * 360 * np.arange(n_bin)
    for track in tracks:
        for i in range(0,len(track.utm)):
            first_p_idx = i - 1
            if first_p_idx < 0:
                first_p_idx = 0
            second_p_idx = i + 1
            if second_p_idx >= len(track.utm):
                second_p_idx = len(track.utm) - 1
            direction = np.array([track.utm[second_p_idx][0] - track.utm[first_p_idx][0],\
                                  track.utm[second_p_idx][1] - track.utm[first_p_idx][1]])
            if np.linalg.norm(direction) > 1.0: 
                """
                Don't count if the length of this vector is smaller than 1.0m, this is to
                remove some direction noise.
                """
                angle = np.angle(direction[0]+direction[1]*1.0j) 
                if angle < 0:
                    angle += 2.0*np.pi
                bin_idx = int(angle / delta_angle)
                results[bin_idx] += 1.0

    results /= sum(results)
    return angles, results

def main():
    if len(sys.argv) != 2:
        print "Error!"
        print "Correct usage is:"
        print "\tpython neighborhood_point_density.py test_case_id"
        return
   
    test_case_id = int(sys.argv[1])
    track_data = "data/beijing_test_road_types/track_%d.dat"%test_case_id
    center = const.TEST_CENTER[test_case_id-1]

    sigma_gps_point = 10.0
    sigma_neighborhood = 100.0
    n_bin = 20

    tracks = gps_track.load_tracks(track_data)

    n_angle_bin = 32
    angles, angle_results = angle_histogram(tracks,
                                    n_angle_bin)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.plot(angles, angle_results, '.-')
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
