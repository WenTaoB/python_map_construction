"""
gps_track: this module contains class definition for Track, and multiple functions that deals with loading,
               saving, and many other related functions.
List:
    - Track: GPS track class.
    - load_tracks: Load GPS tracks from GpsTrack PBF file.
    - extract_tracks_by_region: Extract GPS tracks that touches a rectangular geo-region.
    - is_boundingbox_overlapping: Check if two rectangular geo-boundingbox are overlapping.
    - remove_gaps_in_tracks: Remove large gaps between consecutive GPS points in tracks.
    - save_tracks: Save GPS tracks to a GpsTrack PBF file.
    - visualize_tracks: Visualize GPS tracks using matplotlib.
    - visualize_tracks_on_kml: Generate a kml file of tracks for visualization.

Created by Chen Chen on 07/15/2013
"""
import glob
import re
import os.path
import cPickle

import math
import matplotlib.pyplot as plt
import simplekml
import pyproj

import const

class Track:
    """ GPS track class
        Attributes:
            - utm: a list of tuples, in the format of [(pt1_easting, pt1_northing, pt1_timestamp), ...].
            - bound_box: [(sw_e, sw_n), (ne_e, ne_n)]. Bounding box for this GPS track.
    """

    def __init__(self):
        self.car_id = -1
        self.utm = [] # A tuple: (easting, northing, timestamp, heavy)
        self.bound_box = [(float('inf'), float('inf')), (-float('inf'), -float('inf'))]

    def add_point(self, gps_pt):
        """ Add a GPS point to the track
            Args:
                - gps_pt: a tuple, (easting, northing, timestamp) in UTM format. Timestamp is in epoch, in us.
        """
        self.utm.append(gps_pt)
        if gps_pt[0] < self.bound_box[0][0]:
            self.bound_box = [(gps_pt[0], self.bound_box[0][1]), (self.bound_box[1][0], self.bound_box[1][1])]
        if gps_pt[0] > self.bound_box[1][0]:
            self.bound_box = [(self.bound_box[0][0], self.bound_box[0][1]), (gps_pt[0], self.bound_box[1][1])]
        if gps_pt[1] < self.bound_box[0][1]:
            self.bound_box = [(self.bound_box[0][0], gps_pt[1]), (self.bound_box[1][0], self.bound_box[1][1])]
        if gps_pt[1] > self.bound_box[1][1]:
            self.bound_box = [(self.bound_box[0][0], self.bound_box[0][1]), (self.bound_box[1][0], gps_pt[1])]

    def compute_boundbox(self):
        """ Compute bounding box of this track
        """
        self.bound_box = [(float('inf'), float('inf')), (-float('inf'), -float('inf'))]

        for pt in self.utm:
            if pt[0] < self.bound_box[0][0]:
                self.bound_box = [(pt[0], self.bound_box[0][1]), (self.bound_box[1][0], self.bound_box[1][1])]
            if pt[0] > self.bound_box[1][0]:
                self.bound_box = [(self.bound_box[0][0], self.bound_box[0][1]), (pt[0], self.bound_box[1][1])]
            if pt[1] < self.bound_box[0][1]:
                self.bound_box = [(self.bound_box[0][0], pt[1]), (self.bound_box[1][0], self.bound_box[1][1])]
            if pt[1] > self.bound_box[1][1]:
                self.bound_box = [(self.bound_box[0][0], self.bound_box[0][1]), (self.bound_box[1][0], pt[1])]

def load_tracks(input_filename):
    """ 
    Load GPS tracks from GpsTrack PBF file.
        Args:
            - input_filename: a string, file name of a dumped cPickle file.
        Return:
            - tracks: a list of objects of class Track.
    """
    if not os.path.exists(input_filename):
        print "ERROR! load_tracks() failed:"
        print '\tFile named %s'%input_filename,'does not exists!'
        return []

    print "Loading GPS tracks from %s ..."%input_filename

    with open(input_filename, 'rb') as in_file:
        tracks = cPickle.load(in_file)
            
    print "\tTotally %d tracks are loaded."%len(tracks)
    
    return tracks 

def save_tracks(tracks, out_filename):
    """ This function saves the GPS tracks as a binary file with file name: out_filename
        Args:
            - tracks: a list of objects of class Track.
            - out_filename: a string, e.g., "xx.dat".
    """
    with open(out_filename, 'wb') as out_file:
        cPickle.dump(tracks, out_file, protocol=2)
        
def is_boundingbox_overlapping(bound_box1, bound_box2):
    """ Check if two rectangular boxes are overlapping.
        Args:
            - bound_box1: the first bounding box, a list of tuple: [(sw_e, sw_n), (ne_e, ne_n)]
            - bound_box2: the first bounding box, a list of tuple: [(sw_e, sw_n), (ne_e, ne_n)]
        Return:
            - result: True/False
    """
    easting_list = [bound_box1[0][0], bound_box1[1][0], bound_box2[0][0], bound_box2[1][0]]
    northing_list = [bound_box1[0][1], bound_box1[1][1], bound_box2[0][1], bound_box2[0][1]]
    easting_list.sort()
    northing_list.sort()

    x_inside = False
    y_inside = False

    if easting_list[0] == bound_box2[0][0]:
        if bound_box1[0][0] <= bound_box2[1][0]:
            x_inside = True
    else:
        if bound_box2[0][0] <= bound_box1[1][0]:
            x_inside = True

    if northing_list[0] == bound_box2[0][1]:
        if bound_box1[0][1] <= bound_box2[1][1]:
            y_inside = True
    else:
        if bound_box2[0][1] <= bound_box1[1][1]:
            y_inside = True

    if x_inside and y_inside:
        return True
    else:
        return False

def extract_tracks_by_region_form_dir(input_directory, output_filename, bound_box):
    """ Extract GPS tracks that TOUCHES a rectangular geo-region.
        Args:
            - input_directory: a DIRECTORY which contains all original .pbf tracks (of our GpsTrack format!).
            - output_filename: .pbf formated GPS tracks in the geo region.
            - bound_box: target bounding box, a tuple of format [(sw_e,sw_n), (ne_e), (ne_n)].
        Output:
            - a PBF file with name out_filename
    """
    input_directory = re.sub('\/$', '', input_directory)
    input_directory += '/'
    files = glob.glob(input_directory+'*.pbf')
    if len(files) == 0:
        print "Error! Empty input directory: %s"%input_directory

    tracks = []
    for filename in files:
        print "Now processing file %s"%filename
        candidate_tracks = load_tracks(filename)
        # Filter tracks by bounding box
        for candidate_track in candidate_tracks:
            # Check if the track touches the target box
            is_touching = is_boundingbox_overlapping(candidate_track.bound_box, bound_box)
            if is_touching:
                tracks.append(candidate_track)
    # output file
    save_tracks(tracks, output_filename)
    print "\t%d tracks extracted to file %s"%(len(tracks), output_filename)

def load_tracks_from_directory(input_directory, n_file):
    """ Load GPS tracks from files in the directory.
        Args:
            - input_directory: a DIRECTORY which contains .dat tracks;
            - n_file: number of files to read;

        Return: 
            - tracks.
    """
    input_directory = re.sub('\/$', '', input_directory)
    input_directory += '/'
    files = glob.glob(input_directory+'*.dat')
    if len(files) == 0:
        print "Error! Empty input directory: %s"%input_directory
        sys.exit()

    if len(files) < n_file:
        print "Error! Not enough files in directory: %s"%input_directory
        sys.exit()

    tracks = []
    n_file_counter = 0
    for filename in files:
        print "Now processing file %s"%filename
        tracks_from_file = load_tracks(filename)
        tracks.extend(tracks_from_file)
        n_file_counter += 1
        if n_file_counter >= n_file:
            break

    print "Totally %d tracks extracted from directory %s"%(len(tracks), input_directory)
    return tracks

def extract_tracks_by_region(input_tracks, output_filename, bound_box):
    """ Extract GPS tracks that TOUCHES a rectangular geo-region.
        Args:
            - input_tracks: input track, array of Tracks;
            - output_filename: .dat formated GPS tracks in the geo region;
            - bound_box: target bounding box, a tuple of format [(sw_e,sw_n), (ne_e), (ne_n)].
        Output:
            - Pickle dumped tracks that touches the bound_box.
    """
    
    tracks = []
    for candidate_track in input_tracks:
        is_in_box = False
        new_track = Track()
        for pt in candidate_track.utm:
            if pt[0] >= bound_box[0][0] and pt[0] <= bound_box[1][0]\
               and pt[1] >= bound_box[0][1] and pt[1] <= bound_box[1][1]:
                # Append point
                new_track.add_point(pt)
            else:
                if is_in_box:
                    # End of a track
                    tracks.append(new_track)
                    new_track = Track()
        if len(new_track.utm) != 0:
            tracks.append(new_track)
    # output file
    save_tracks(tracks, output_filename)
    
    print "\t%d tracks extracted to file %s"%(len(tracks), output_filename)
    return tracks

def remove_gaps_in_tracks(tracks, 
                          max_delta_t, 
                          max_delta_d,
                          min_recording_pt_count, 
                          bound_box):
    """ Remove gaps between adjacent GPS points in tracks. Gaps that are larger than max_delta_dist will be 
        removed.
        Args:
            tracks: a list of objects of class Track (defined in track.py)
            max_delta_t: maximum time between adjacent GPS points in a track. If the diff is 
                            larger than max_delta_t, a new track will be created
            max_delta_t: maximum distance between adjacent GPS points
            min_recording_pt_count: if a track has less GPS points than this value, it will not be recorded
            bound_box: a tuple of format [(sw_e, se_n), (ne_e, ne_n)] the recording bounding box. If the track 
                       does not intersect with this bounding box, it will not be recorded either.
        Return:
            new_tracks: a list of objects of class Track
    """

    new_tracks = []

    for track in tracks:
        pt_index = 1

        new_track = Track()
        new_track.add_point(track.utm[0])
        
        while pt_index < len(track.utm):
            #delta_t = track.utm[pt_index][2] - track.utm[pt_index-1][2]
            delta_t = 5
            delta_d = math.sqrt((track.utm[pt_index][0] - track.utm[pt_index-1][0])**2 +\
                                (track.utm[pt_index][1] - track.utm[pt_index-1][1])**2)
            if delta_t <= max_delta_t and delta_d <= max_delta_d:
                # Record this point
                new_track.add_point(track.utm[pt_index])
            else:
                is_touching = is_boundingbox_overlapping(new_track.bound_box, bound_box)
                if len(new_track.utm) >= min_recording_pt_count and is_touching:
                    new_tracks.append(new_track)
                
                # Start a new track
                new_track = Track()
                new_track.add_point(track.utm[pt_index])

            pt_index += 1

        is_touching = is_boundingbox_overlapping(new_track.bound_box, bound_box)
 
        if len(new_track.utm) >= min_recording_pt_count and is_touching:
            new_tracks.append(new_track)

    return new_tracks 

def visualize_tracks(tracks, bound_box = [(-1,-1), (-1,-1)], ax = -1, style='.'):
    """ Visualize GPS tracks using matplotlib.
        Args:
            - tracks: a list of Track objects.
            - bound_box: OPTIONAL, visualization bounding box.
    """

    print "GPS track visualization: %d tracks."%len(tracks)
    direct_display_mode = False
    if ax == -1:
        direct_display_mode = True
        fig = plt.figure(figsize = const.figsize)
        ax = fig.add_subplot(111, aspect='equal')

    count = 0
    for track in tracks:
        ax.plot([utm[0] for utm in track.utm], 
                [utm[1] for utm in track.utm], 
                style,
                linewidth=0.2)
                #, color=color)

    if bound_box[0][0] != -1 and bound_box[0][1] != -1 and bound_box[1][0]!= -1 and bound_box[1][1]!= -1:
        ax.set_xlim([bound_box[0][0], bound_box[1][0]])
        ax.set_ylim([bound_box[0][1], bound_box[1][1]])
    else:
        ax.autoscale()

    if direct_display_mode:
        plt.show()

def visualize_tracks_around_center(tracks,
                                   center,
                                   radius,
                                   ax = -1,
                                   style = '.'):
    """
        Visualize GPS tracks around a center points. Bound box will be centered at center, 
        with length 2 * radius.
        Args:
            - tracks: a list of Track;
            - center: (easting, northing);
            - radius: in meters.
    """
    direct_display_mode = False
    if ax == -1:
        direct_display_mode = True
        fig = plt.figure(figsize = const.figsize)
        ax = fig.add_subplot(111, aspect='equal')

    for track in tracks:
        ax.plot([utm[0] for utm in track.utm], [utm[1] for utm in track.utm], style)

    ax.set_xlim([center[0]-radius, center[0]+radius])
    ax.set_ylim([center[1]-radius, center[1]+radius])

    if direct_display_mode:
        plt.show()

def visualize_tracks_on_kml(tracks, out_filename):
    """ Generate a kml file containing all tracks as linestring
            Args:
                - tracks: a list of Track objects.
                - out_filename: a string, the output file name.
            Output:
                - a kml file with name "out_filename".
    """
    utm_projector = pyproj.Proj(proj='utm', zone=50, south=False, ellps='WGS84')
    
    kml = simplekml.Kml()
    colors = ["501400FF", "5014F0FF", "5078FF00", "50FF7800", "50FF78F0", "50F0FF14", "5078003C"]
    count = 0
    for track in tracks:
        coords = []
        for pt in track.utm:
            lon, lat = utm_projector(pt[0], pt[1], inverse=True)
            coords.append((lon, lat))
        line = kml.newlinestring()

        color_index = count%7
        line.style.linestyle.color = colors[color_index]
        line.style.linestyle.width = 4
        line.coords = coords
        count += 1
    kml.save(out_filename)
