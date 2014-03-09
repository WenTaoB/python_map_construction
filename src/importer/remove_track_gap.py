# -*- coding: utf-8 -*-
"""
Created on Sun Oct 06 21:40:34 2013

Extract GPS points defined by a geo-region.

@author: ChenChen
"""
import sys
import cPickle

import matplotlib.pyplot as plt

import gps_track
import const

def main():
    if len(sys.argv) != 3:
        print "Error! Correct usage is:"
        print "\tpython remove_gaps_in_tracks.py [orig_track_file_name] [output_track_file_name]"
        return

    with open(sys.argv[1], "rb") as fin:
        tracks = cPickle.load(fin)
    
    new_tracks = gps_track.remove_gaps_in_tracks(tracks, 
                                                 65, 
                                                 500, 
                                                 10, 
                                                 [const.RANGE_SW, const.RANGE_NE])
    gps_track.save_tracks(new_tracks, sys.argv[2]) 

    print "%d tracks saved to %s."%(len(new_tracks), sys.argv[2])

        
if __name__ == "__main__":
    sys.exit(main())
