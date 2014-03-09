#!/usr/bin/env python

import sys
import cPickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

import osm_for_drawing
import const

def main():
    if len(sys.argv) != 2:
        print "Error!"
        return
    
    with open(sys.argv[1], 'rb') as in_file:
        tracks = cPickle.load(in_file)

    for i in range(0, len(tracks)):
        track = tracks[i]
        file_name = "trips/trip_%d.txt"%(i+1)
        with open(file_name, 'w') as fout:
            for pt in track.utm:
                fout.write("%.2f %.2f %.2f\n"%(pt[0],pt[1],pt[2]))

if __name__ == "__main__":
    sys.exit(main())
