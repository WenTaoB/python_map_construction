#!/usr/bin/env python

"""
Created on Wed Oct 09 16:56:33 2013
@author: ChenChen
"""
import sys
import re
import os
import cPickle
from optparse import OptionParser

import gps_track
import const

def main():
    parser = OptionParser()
    parser.add_option("-t","--tracks", dest="track_data", help="Input GPS track filename.", type="string", metavar="Tracks")
    parser.add_option("-o","--output_dir", dest="output_dir", help="Output directory.", type="string", metavar="Output_dir")
    parser.add_option("-m", "--mode", dest="output_mode", type="int", help="Output mode: 0: default output mode, which agrees with mapconstructionportal.org trip format; 1: output agrees with James2012 data format.", default=0)
    (options, args) = parser.parse_args()

    if not options.track_data:
        parser.error("No input track data file not found!")
    if not options.output_dir:
        parser.error("Output directory is not specified!")

    if not os.path.exists(options.output_dir):
        parser.error("Output directory does not exist! Please create it first!")

    if options.output_mode != 0 and options.output_mode != 1:
        parser.error("Unsupported output mode. (Output mode has to be 0 or 1)")
    
    if options.output_mode == 1:
        print "WARNING: please check if you are using the correct utm_projector."
    
    output_directory = re.sub('\/$', '', options.output_dir)
    output_directory += "/"

    tracks = gps_track.load_tracks(options.track_data)

    # Write to file
    for i in range(0, len(tracks)):
        output_filename = output_directory + "trip_%d.txt"%i
        track = tracks[i]

        if len(track.utm) <2:
            continue

        if options.output_mode == 0:
            with open(output_filename, "w") as fout:
                for utm in track.utm:
                    utm_time = utm[2] / 1e6
                    fout.write("%.2f %.2f %.2f\n"%(utm[0], utm[1], utm_time))
        else:
            with open(output_filename, "w") as fout:
                pt_id = 0
                for utm in track.utm:
                    lon, lat = const.SF_utm_projector(utm[0], utm[1], inverse=True)
                    utm_time = utm[2] / 1e6
                    if pt_id == 0:
                        fout.write("%d,%.6f,%.6f,%.1f,None,%d\n"%(pt_id, lat, lon, utm_time, pt_id+1))
                    elif pt_id < len(track.utm) - 1:
                        fout.write("%d,%.6f,%.6f,%.1f,%d,%d\n"%(pt_id, lat, lon, utm_time, pt_id-1, pt_id+1))
                    else:
                        fout.write("%d,%.6f,%.6f,%.1f,%d,None\n"%(pt_id, lat, lon, utm_time, pt_id-1))
                    pt_id += 1

if __name__ == "__main__":
    sys.exit(main())
