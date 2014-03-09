#!/usr/bin/env python
"""
Created on 01-22-2014

@author: ChenChen
"""
import glob
import re
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
        print "\tpython extract_tracks_in_region.py [input_directory] [out_track_file]"
        return
 
    # Index for the test region, 0..9
    index = 4
    BBOX_SW = const.BB_SW[index]
    BBOX_NE = const.BB_NE[index]

    input_directory = re.sub('\/$', '', sys.argv[1])
    input_directory += '/'
    files = glob.glob(input_directory+'*.dat')
    if len(files) == 0:
        print "Error! Empty input directory: %s"%input_directory

    extracted_tracks = []

    count = 0
    MIN_PT_COUNT = 4
    for filename in files:
        print "Now processing ",filename
        input_tracks = gps_track.load_tracks(filename)
        for track in input_tracks:
            # Iterate over its point
            is_recording = False
            recorded_track = gps_track.Track()
            for pt_idx in range(0, len(track.utm)):
                # Check if the point falls inside the bounding box
                if track.utm[pt_idx][0] >= BBOX_SW[0] and \
                   track.utm[pt_idx][0] <= BBOX_NE[0] and \
                   track.utm[pt_idx][1] >= BBOX_SW[1] and \
                   track.utm[pt_idx][1] <= BBOX_NE[1]:
                       if not is_recording:
                           # Start recording
                           is_recording = True
                           recorded_track.car_id = track.car_id
                           if pt_idx > 0:
                               recorded_track.add_point(track.utm[pt_idx-1]) 
                               recorded_track.add_point(track.utm[pt_idx])
                       else:
                           # Append point
                           recorded_track.add_point(track.utm[pt_idx])
                else:
                    # Point is outside the bounding box
                    if is_recording:
                        # Stop recording
                        is_recording = False
                        recorded_track.add_point(track.utm[pt_idx])
                        if len(recorded_track.utm) >= MIN_PT_COUNT:
                            # Save the recorded track
                            extracted_tracks.append(recorded_track)
                        recorded_track = gps_track.Track()

        count += 1

        if count == 4:
            break
    # Visualize extracted GPS tracks
    print "%d tracks extracted"%len(extracted_tracks)
    gps_track.visualize_tracks(extracted_tracks, bound_box = [BBOX_SW, BBOX_NE], style='.')
    gps_track.save_tracks(extracted_tracks, sys.argv[2]) 

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
