#!/usr/bin/env python

"""
Road Detector

Created on Wed Sep 25 08:51:53 2013

@author: ChenChen
"""

import sys
import random
import cPickle

import scipy.spatial
import matplotlib.pyplot as plt

import gps_track
import const

def compute_hist(query_loc, radius, qtree):
    """ 
    Comput neighborhoot point density histogram.
        Args:
            - query_loc: a tuple, (easting, northin) of the query point;
            - radius: a list of radiuses, i.e., the discrete sampling point;
            - qtree: a KDTree.
        Return:
            - hist_result: a list, the same size as radius, the density at each radius. Results are normalized by \
                            the total number of points in the largest circle.
    """
    hist_result = []
    for r in radius:
        nearby_point = qtree.query_ball_point(query_loc, r)
        hist_result.append(float(len(nearby_point)))

    index = len(hist_result) - 1
    while index != 1:
        hist_result[index] -= hist_result[index-1]
        index -= 1
    
    total_point = sum(hist_result)
    if total_point != 0:
        for index in range(0, len(hist_result)):
            hist_result[index] /= total_point
    return hist_result

def main():
    if len(sys.argv) != 3:
        print "Error! Correct usage is:"
        print "\tpython road_detector.py [input_track.dat]"
        return
    tracks = gps_track.load_tracks(sys.argv[1])
    tracks1 = gps_track.load_tracks(sys.argv[2])
    tracks.extend(tracks1)

    RANGE_SW = (446000, 4421450)
    RANGE_NE = (451000, 4426450)

    point_collection = []
    for track in tracks:
        for pt in track.utm:
            if pt[0] <= RANGE_NE[0] and pt[0] >= RANGE_SW[0]:
                if pt[1] <= RANGE_NE[1] and pt[1] >= RANGE_SW[1]:
                    point_collection.append((pt[0], pt[1]))
    print "There are %d GPS points."%len(point_collection)
    qtree = scipy.spatial.KDTree(point_collection)
    print "Quad tree completed."
    
    training_loc = []
    training_feature = []
    count = 0
    query_radius = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    
    while True:
        #rand_easting = random.randint(RANGE_SW[0]+500, RANGE_NE[0]-500)
        #rand_northing = random.randint(RANGE_SW[1]+500, RANGE_NE[1]-500)
        #query_loc = (rand_easting, rand_northing)
        while True:
            ind = random.randint(0, len(point_collection)-1)
            query_loc = point_collection[ind]
            if query_loc[0] <= RANGE_NE[0]-500 and query_loc[0] >= RANGE_SW[0]+500:
                if query_loc[1] <= RANGE_NE[1]-500 and query_loc[1] >= RANGE_SW[1]+500:
                    break   
        
        training_loc.append(query_loc)
        print "Query location is: ", query_loc
        fig = plt.figure(figsize=(16,16))
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.plot([p[0] for p in point_collection],
                 [p[1] for p in point_collection],
                 '.', 
                 color='gray')
        ax1.plot(query_loc[0], query_loc[1], 'r+', markersize=12)
        ax1.set_xlim([RANGE_SW[0], RANGE_NE[0]])
        ax1.set_ylim([RANGE_SW[1], RANGE_NE[1]])
        hist_result = compute_hist(query_loc, query_radius, qtree)    
        training_feature.append(list(hist_result))
        out_filename = "tmp\\"+"%d"%count + ".png"
        plt.savefig(out_filename)
        plt.close(fig)
        count += 1 

        if count == 100:
            break
        
    with open("training_loc.dat", "wb") as fout:
        cPickle.dump(training_loc, fout, protocol=2)
    with open("training_feature.dat", "wb") as fout:
        cPickle.dump(training_feature, fout, protocol=2)
#    with open("training_label.dat", "wb") as fout:
#        cPickle.dump(training_label, fout, protocol=2)

if __name__ == "__main__":
    sys.exit(main())
