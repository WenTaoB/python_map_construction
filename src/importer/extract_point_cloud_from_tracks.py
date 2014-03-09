#!/usr/bin/env python
"""
Using a unified grid to rasterize GPS tracks.
"""

import sys
import cPickle

import gps_track
from point_cloud import PointCloud
import const

def main():
    if len(sys.argv) != 3:
        print "ERROR! Correct usage is"
        print "\tpython extract_point_cloud_from_tracks.py [track_file.dat] [output_point_cloud.dat]"
        return

    tracks = gps_track.load_tracks(sys.argv[1])

    # Target location and region radius
    #LOC = (447772, 4424300)
    #R = 500

    #LOC = (446458, 4422150)
    #R = 500

    LOC = (551281, 4180430) # San Francisco
    R = 500

    point_cloud = PointCloud()
    point_cloud.extract_point_cloud(tracks, LOC, R)
    point_cloud.visualize_point_cloud(LOC, R)

    point_cloud.save(sys.argv[2])

if __name__ == "__main__":
    sys.exit(main())
