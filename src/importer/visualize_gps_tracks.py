#!/usr/bin/env python
"""
Visualize GPS tracks.
Created on Tue Sep 24 09:27:15 2013

@author: ChenChen
"""

import sys
import cPickle

import gps_track
import const

def main():
    with open(sys.argv[1], "rb") as fin:
        tracks = cPickle.load(fin)
#    with open("test1.dat", "wb") as fout:
#        cPickle.dump(tracks, fout, protocol=0)
    print "There are %d tracks"%len(tracks)

    #LOC = (447772, 4424300)
    #R = 500

    LOC = (446458, 4422150)
    R = 500

    gps_track.visualize_tracks(tracks, 
                               style='.b',
                               bound_box = [(LOC[0]-R, LOC[1]-R), (LOC[0]+R, LOC[1]+R)])

if __name__ == "__main__":
    sys.exit(main())
