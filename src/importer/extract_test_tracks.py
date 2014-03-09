#!/usr/bin/env python
"""
Created on Mon Oct 07 17:19:49 2013

@author: ChenChen
"""

import sys
import cPickle
import math

import matplotlib.pyplot as plt
from matplotlib import cm as CM

import gps_track
import const

def main():
    if len(sys.argv) != 3:
        print "Error! Correct usage is:"
        print "\tpython extract_test_tracks.py [input_track_file] [out_track_file]"
        return
    
    input_tracks = gps_track.load_tracks(sys.argv[1])
    output_tracks = gps_track.extract_tracks_by_region(input_tracks, 
                                                       sys.argv[2], 
                                                        (const.SF_small_RANGE_SW, const.SF_small_RANGE_NE))
    
    # Visualization
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    for track in output_tracks:
        ax.plot([pt[0] for pt in track.utm],
                [pt[1] for pt in track.utm],
                '.-'
                )
    plt.show()
    
if __name__ == "__main__":
    sys.exit(main())
