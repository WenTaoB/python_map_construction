#!/usr/bin/env python

import os.path
import struct
import cPickle

import pyproj

import const
import gps_track

def unpack_log_record_pbf(input_filename, output_filename = ""):
    """ Unpack LogRecord protobuf from a PBF file, convert it from lon/lat to UTM format and convert original 
           CFAbsoluteTime to UTC.

            Args:
                - input_filename: a str file name. The file must be in LogRecord protobuf format.
                - min_n_point: minimum number of GPS points to save a track. If a track have GPS points < 
                               min_n_point, it will not be recorded.
                - output_filename: a str file name. 
            Output:
                - A file with name output_filename. It's a PBF file, where each track will be a binary protobuf
                  of gps_track_pb2.GpsTrack().

            Created by Chen Chen on 07/08/2013
    """

    tracks = []
    utm_projector = pyproj.Proj(proj='utm', zone=50, south=False, ellps='WGS84')
    if not os.path.exists(input_filename):
        print 'File named %s'%input_filename,'does not exists!'
        print 'Usage: read_pbf(input_filename, min_n_point, output_filename)'
        return tracks 
    track_count = 0
    with open(input_filename, 'rb') as in_file:
        a_track = gps_track.Track()
        last_car_id = -1
        while True:
            data_stream = in_file.read(28)
            if len(data_stream) < 28:
                break
            data_field = struct.unpack("iiiiiii", data_stream)            
            car_id = data_field[0]

            if car_id != last_car_id:
                track_count += 1
                if last_car_id != -1:
                    tracks.append(a_track)
                    a_track = gps_track.Track()
                last_car_id = car_id
                a_track.car_id = car_id
                
            utc_time = data_field[1] + const.CF_ABSOLUTE_TIME_TO_UTC_OFFSET
            lon = float(data_field[3]) / 1e5
            lat = float(data_field[2]) / 1e5
            (easting, northing) = utm_projector(lon, lat)
            a_track.add_point((easting, northing, utc_time, data_field[6]))
            
    print "Totally %d effective tracks are recorded from %s"%(len(tracks), input_filename)

    if len(output_filename) > 0:
        with open(output_filename, 'wb') as out_file:
            cPickle.dump(tracks, out_file, protocol=2)
