#!/usr/bin/env python
"""
Created on Thu Sep 26 14:20:00 2013

@author: ChenChen
"""

import sys
import cPickle
import random

import sklearn.svm
import scipy.spatial
import matplotlib.pyplot as plt

import gps_track
import road_detector

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
    
    # Training SVM
    with open("training_loc_0.dat", "rb") as fin:
        training_loc = cPickle.load(fin)
    with open("training_feature_0.dat", "rb") as fin:
        training_feature = cPickle.load(fin)
    with open("training_loc_1.dat", "rb") as fin:
        training_loc1 = cPickle.load(fin)
    with open("training_feature_1.dat", "rb") as fin:
        training_feature1 = cPickle.load(fin)
    training_loc.extend(training_loc1)
    training_feature.extend(training_feature1)
    with open("training_label.dat", "rb") as fin:
        training_label = cPickle.load(fin)
    svc = sklearn.svm.SVC(kernel='sigmoid').fit(training_feature, training_label)
    
    # Make Prediction
    road_point = []
    non_road_point = []
    query_radius = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    count = 0

    # Start Prediction
    print "Start prediction!"
    for pt in point_collection:
        rand_easting = random.randint(RANGE_SW[0]+500, RANGE_NE[0]-500)
        rand_northing = random.randint(RANGE_SW[1]+500, RANGE_NE[1]-500)
        pt = (rand_easting, rand_northing)
        
#        if pt[0] <= RANGE_NE[0]-500 and pt[0] >= RANGE_SW[0]+500 \
#                and pt[1] <= RANGE_NE[1]-500 and pt[1] >= RANGE_SW[1]+500:
#            print count
        hist_result = road_detector.compute_hist(pt, query_radius, qtree)
        prediction = svc.predict([hist_result])
        if prediction[0] == 0:
            non_road_point.append(pt)
        elif prediction[0] == 1:
            road_point.append(pt)
        else:
            print "Warning! Invalid prediction!"
        count += 1
        if count % 500 == 0:
            print "Now at ",count
            break
    print "There are %d road point, %d non-road point."%(len(road_point), len(non_road_point))
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot([p[0] for p in point_collection], 
            [p[1] for p in point_collection],
            '.',
            color='gray')
    ax.plot([p[0] for p in road_point], 
            [p[1] for p in road_point],
            '.',
            color='red')
    ax.plot([p[0] for p in non_road_point], 
            [p[1] for p in non_road_point],
            '.',
            color='blue')            
    ax.set_xlim([RANGE_SW[0], RANGE_NE[0]])
    ax.set_ylim([RANGE_SW[1], RANGE_NE[1]])   
    plt.show()
    
if __name__ == "__main__":
    sys.exit(main())

