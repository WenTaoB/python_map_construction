#!/usr/bin/env python
"""
    Functions that extract GPS tracks in a local neighborhood around a Geo location.
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

def circular_neighborhood(tracks,
                          center,
                          R_inner,
                          R_outer):
    """
        Extract persistent GPS tracks around the center point.
        Args:
            - tracks: a list of Track;
            - center: (easting, northing);
            - R_inner: inner circle radius;
            - R_outer: outer circle radius.
        Return:
            - neighborhood_tracks.
    """
    neighborhood_tracks = []

    # Indexing each track points
    idx = 0
    track_idx_inside_outer_circle = {}
    for track in tracks:
        for utm in track.utm:
            # Compute distance to center
            dist = np.sqrt((utm[0]-center[0])**2 + (utm[1]-center[1])**2)
            if dist <= R_outer:
                track_idx_inside_outer_circle[idx] = 1
                break
        idx += 1
    # Select only the track that intersect with the inner circle
    for track_idx in track_idx_inside_outer_circle.keys():
        track = tracks[track_idx]
        for pt_idx in range(0, len(track.utm)-1):
            vec = np.array([track.utm[pt_idx+1][0] - track.utm[pt_idx][0],\
                            track.utm[pt_idx+1][1] - track.utm[pt_idx][0],\
                            0.0])
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1.0:
                # Neglect if the two consecutive points are too close
                continue
            vec /= vec_norm
            vec_cross = np.cross(vec, [-1*vec[1], vec[0], 0.0])
            vec_center = [center[0] - track.utm[pt_idx][0],\
                          center[1] - track.utm[pt_idx][1],\
                          0.0]
            dist = abs(np.dot(vec_center, vec_cross))

            if dist <= R_inner:
                neighborhood_tracks.append(track)
                break
    print "\nResults:%d tracks in neighborhood around"%(len(neighborhood_tracks)),center
    return neighborhood_tracks

def main():
    tracks = gps_track.load_tracks_from_directory(sys.argv[1], 7)
    gps_track.save_tracks(tracks, sys.argv[2])
    center = (440235, 4423320)
    neighborhood_tracks = circular_neighborhood(tracks, 
                                                center, 
                                                50, 
                                                100)

    # Visualization
    gps_track.visualize_tracks_around_center(tracks,
                                             center,
                                             500)

if __name__ == "__main__":
    sys.exit(main())
