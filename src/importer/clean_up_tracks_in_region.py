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
    output_tracks = gps_track.remove_gaps_in_tracks(input_tracks, 10, 300, 20,
                                                    (const.SF_RANGE_SW, const.SF_RANGE_NE))
    print "There are %d extracted tracks."%len(output_tracks)
    gps_track.save_tracks(output_tracks, sys.argv[2])
    
if __name__ == "__main__":
    sys.exit(main())
