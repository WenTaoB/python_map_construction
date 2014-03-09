#!/usr/bin/env python
"""
    unpack_log_record_in_dir.py

    Functionality: unpack all LogRecord PBF files in a directory.
        Invoke example:
            python log_record_to_gps_track_file.py [input_filename] [output_filename] [min_pt_count].
        Args:
            - input_filename: a str, the input LogRecord PBF file name.
            - output_filename: a str, the output GpsTrack PBF file. See gps_track_pb2 for definition.
            - min_pt_count: minimum number of GPS points in order to record a track. If a track has GPS points
                            smaller than this number, it will not be recorded.

    Created by Chen Chen on 07/08/2013.
"""
import os
import sys
import re
import glob

import unpack_raw_gps_track

def main():
    if len(sys.argv) != 3:
        Usage()
        return
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    input_directory = re.sub('\\$', '', input_directory)
    input_directory += '\\'
    output_directory = re.sub('\\$', '', output_directory)
    output_directory += '\\'
    files = glob.glob(input_directory+'*.dat')
    if len(files) == 0:
        print "Error! Empty input directory: %s"%input_directory
    
    for filename in files:
        print "Now processing %s"%filename
        out_filename = os.path.basename(filename)
        out_filename = re.sub('\.dat$', '', out_filename)
        out_filename = output_directory + out_filename
        out_filename += '.dat'        
        unpack_raw_gps_track.unpack_log_record_pbf(filename, out_filename)
    
def Usage():
    print 'Correct usage: python log_record_to_gps_track_file.py [input_directory] [output_directory]'

if __name__ == "__main__":
    sys.exit(main())
