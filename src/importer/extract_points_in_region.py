# -*- coding: utf-8 -*-
"""
Created on Sun Oct 06 21:40:34 2013

Extract GPS points defined by a geo-region.

@author: ChenChen
"""
import sys
import cPickle

import matplotlib.pyplot as plt

import const

def extract_GPS_point(tracks, bounding_box):
    """
        Args:
            - tracks: a list of Track objects.
            - bounding_box: [(sw_e, sw_n), (ne_e, ne_n)].
        Return:
            - extracted_points: a list of points: [(p1_e, p1_n), ...].
    """
    extracted_points = []
    for track in tracks:
        for pt in track.utm:
            if pt[0] >= bounding_box[0][0] and pt[0] <= bounding_box[1][0] \
                and pt[1] >= bounding_box[0][1] and pt[1] <= bounding_box[1][1]:
                    extracted_points.append(pt)
    return extracted_points

def main():
    if len(sys.argv) != 3:
        print "Error! Correct usage is:"
        print "\tpython extract_points_in_region.py [track_file.dat] [output_filename.dat]"
        return
    with open(sys.argv[1], "rb") as fin:
        tracks = cPickle.load(fin)
        
    extracted_points = extract_GPS_point(tracks, [const.RANGE_SW, const.RANGE_NE])

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')    
    ax.plot([pt[0] for pt in extracted_points], 
            [pt[1] for pt in extracted_points],
            '.')
    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])
    plt.show()
    with open(sys.argv[2], 'wb') as fout:
        cPickle.dump(extracted_points, fout, protocol=2)
    
if __name__ == "__main__":
    sys.exit(main())