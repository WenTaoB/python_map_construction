#!/usr/bin/env python
"""
Created on Mon Nov 25, 2013

@author: ChenChen

Convert our GPS track data to data that can be accepted by map_inference_algorithms.

"""

import sys
import cPickle
import copy
import re
import pyproj

import matplotlib.pyplot as plt
from matplotlib import cm as CM
import numpy as np

import gps_track
import const

def main():
    if len(sys.argv) != 3:
        print "Error! Correct usage is:"
        print "\tpython track_converter.py [input_track_file] [output_directory_name]."
        return

    output_directory = sys.argv[2]
    output_directory = re.sub('\/$', '', output_directory)
    output_directory += '/'

    utm_projector = pyproj.Proj(proj='utm', zone=10, south=False, ellps='WGS84')
    tracks = gps_track.load_tracks(sys.argv[1])
    
    count = 0
    loc_id = 0
    for track in tracks:
        output_filename = output_directory + "trip_%d.txt"%(count)

        f = open(output_filename, 'w')

        for pt_idx in range(0, len(track.utm)):
            pt = track.utm[pt_idx]
            lon, lat = utm_projector(pt[0], pt[1], inverse=True)
            time = pt[2] / 1e6
            prev_id = 'None'
            next_id = 'None'
            cur_id = "%d"%loc_id
            if pt_idx > 0:
                prev_id = "%d"%(loc_id - 1)
            if pt_idx < len(track.utm) - 1:
                next_id = "%d"%(loc_id + 1)

            f.write("%s,%.6f,%.6f,%.1f,%s,%s\n"%(cur_id, lat, lon, time, prev_id, next_id))
            loc_id += 1
            
        f.close()
        count += 1

if __name__ == "__main__":
    sys.exit(main())
