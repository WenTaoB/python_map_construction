#!/usr/bin/env python
"""
Extract point cloud from a collection of GPS tracks.
"""

import sys
import cPickle
from optparse import OptionParser

import numpy as np
import gps_track
from point_cloud import PointCloud
import const

def main():
    parser = OptionParser()

    parser.add_option("-t", "--track", dest="track_file", help="GPS track file name", metavar="TRACK_FILE", type="string")
    parser.add_option("-o", "--output", dest="output_filename", help="Output point cloud file name, e.g., output_point_cloud.dat", type="string")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)

    (options, args) = parser.parse_args()

    if not options.track_file:
        parser.error("Track file not given.")
    if not options.output_filename:
        parser.error("Output pointcloud filename not given.")
   
    R = const.R
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    tracks = gps_track.load_tracks(options.track_file)

    point_cloud = PointCloud(np.array([]), np.array([]))
    point_cloud.extract_point_cloud(tracks, LOC, R)
    point_cloud.visualize_point_cloud(LOC, R)
    point_cloud.save(options.output_filename)

if __name__ == "__main__":
    sys.exit(main())
