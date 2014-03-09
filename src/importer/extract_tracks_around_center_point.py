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
from matplotlib.collections import LineCollection

import gps_track
import const

def extract_tracks(input_directory,
                     center,
                     R,
                     BBOX_WIDTH,
                     BBOX_SW,
                     BBOX_NE,
                     MIN_PT_COUNT,
                     N_DAY):
    input_directory = re.sub('\/$', '', input_directory)
    input_directory += '/'
    files = glob.glob(input_directory+'*.dat')
    if len(files) == 0:
        print "Error! Empty input directory: %s"%input_directory

    extracted_tracks = []

    count = 0
    for filename in files:
        print "Now processing ",filename
        input_tracks = gps_track.load_tracks(filename)
        for track in input_tracks:
            # Iterate over its point
            to_record = False
            for pt_idx in range(0, len(track.utm)):
                # Check if the point falls inside the bounding box
                delta_e = track.utm[pt_idx][0] - center[0]
                delta_n = track.utm[pt_idx][1] - center[1]
                dist = math.sqrt(delta_e**2 + delta_n**2)
                if dist <= R:
                    to_record = True
                    break
            if not to_record:
                continue
            recorded_track = gps_track.Track()
            is_recording = False
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

        if count == N_DAY:
            break
    return extracted_tracks

def extract_tracks_from_file(input_filename,
                             center,
                             R,
                             BBOX_WIDTH,
                             BBOX_SW,
                             BBOX_NE,
                             MIN_PT_COUNT):
    files = [input_filename]

    extracted_tracks = []

    count = 0
    for filename in files:
        print "Now processing ",filename
        input_tracks = gps_track.load_tracks(filename)
        for track in input_tracks:
            # Iterate over its point
            to_record = False
            for pt_idx in range(0, len(track.utm)):
                # Check if the point falls inside the bounding box
                delta_e = track.utm[pt_idx][0] - center[0]
                delta_n = track.utm[pt_idx][1] - center[1]
                dist = math.sqrt(delta_e**2 + delta_n**2)
                if dist <= R:
                    to_record = True
                    break
            if not to_record:
                continue
            recorded_track = gps_track.Track()
            is_recording = False
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
    return extracted_tracks

def save_image(extracted_tracks,
               output_directory,
               center,
               R,
               BBOX_WIDTH,
               BBOX_SW,
               BBOX_NE,
               osm_filename_prefix,
               track_filename_prefix,
               cur_id,
               osm_file):
    output_directory = re.sub('\/$', '', output_directory)
    output_directory += '/'
    osm_output_filename = output_directory + osm_filename_prefix + "_%d"%cur_id + ".png"
    track_output_filename = output_directory + track_filename_prefix + "_%d"%cur_id + ".png"

    # Draw tracks
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, aspect='equal')
    for track in extracted_tracks:
        ax.plot([pt[0] for pt in track.utm],
                [pt[1] for pt in track.utm],
                '.')
    ax.set_xlim([BBOX_SW[0], BBOX_NE[0]])
    ax.set_ylim([BBOX_SW[1], BBOX_NE[1]])
    plt.savefig(track_output_filename, dpi=100)
    plt.clf

    # Draw osm
    #with open(osm_file, 'rb') as in_file:
    #    drawing_osm = cPickle.load(in_file)
    #fig = plt.figure()
    #fig.set_size_inches(10, 10)
    #ax = fig.add_subplot(111, aspect='equal')
    #easting, northing = drawing_osm.node_list()
    #edge_list = drawing_osm.edge_list()
    ##print edge_list
    #edge_collection = LineCollection(edge_list, colors='gray', linewidths=2)
    #ax.add_collection(edge_collection)

    #ax.set_xlim([BBOX_SW[0], BBOX_NE[0]])
    #ax.set_ylim([BBOX_SW[1], BBOX_NE[1]])

    #plt.savefig(osm_output_filename, dpi=100)
    #plt.clf

def main():
    if len(sys.argv) != 5:
        print "Error! Correct usage is:"
        print "\tpython extract_tracks_around_center_point.py [input_directory] [osm_for_drawing.dat] [out_track_file_directory] [output_image_directory]"
        return
    """
        R: query radius, in meters, track will be recorded once it has a point 
           falls into this ball centered at the center point.
    """
    BBOX_WIDTH = 500
    R = BBOX_WIDTH
    cur_id = 1
    #center = (447772, 4424300)
    #center = (446458, 4422150)
    center = (551281, 4180430) # San Francisco

    BBOX_SW = (center[0]-BBOX_WIDTH, center[1]-BBOX_WIDTH)
    BBOX_NE = (center[0]+BBOX_WIDTH, center[1]+BBOX_WIDTH)
    MIN_PT_COUNT = 4
    N_DAY = 3

    #extracted_tracks = extract_tracks(sys.argv[1],
    #                                  center,
    #                                  R, 
    #                                  BBOX_WIDTH, 
    #                                  BBOX_SW, 
    #                                  BBOX_NE, 
    #                                  MIN_PT_COUNT,
    #                                  N_DAY)

    extracted_tracks = extract_tracks_from_file(sys.argv[1],
                                                center,
                                                R, 
                                                BBOX_WIDTH, 
                                                BBOX_SW, 
                                                BBOX_NE, 
                                                MIN_PT_COUNT)

    print "%d tracks extracted"%len(extracted_tracks)
    
    track_file_directory = re.sub('\/$', '', sys.argv[3])
    track_file_directory += '/'
    track_filename = track_file_directory + "SF_track" + "_%d"%cur_id + ".dat"
    gps_track.save_tracks(extracted_tracks, track_filename) 
    save_image(extracted_tracks,
               sys.argv[4],
               center,
               R,
               BBOX_WIDTH,
               BBOX_SW,
               BBOX_NE,
               "osm",
               "track_point",
               cur_id,
               sys.argv[2])
    # Visualize extracted GPS tracks
    gps_track.visualize_tracks(extracted_tracks, bound_box = [BBOX_SW, BBOX_NE], style='.')
    return

if __name__ == "__main__":
    sys.exit(main())
