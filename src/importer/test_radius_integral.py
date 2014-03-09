# -*- coding: utf-8 -*-
"""
Created on Sun Oct 06 22:02:23 2013

@author: ChenChen
"""

import sys
import cPickle

import scipy.spatial
import matplotlib.pyplot as plt

import const

def main():
    if len(sys.argv) != 2:
        print "ERROR! Correct usage is:"
        print "\tpython test_radius_integral.py [gps_point_collection.dat]"
        return
    # Load GPS points
    with open(sys.argv[1], "rb") as fin:
        point_collection = cPickle.load(fin)
    
    RADIUS = 50
    THRESHOLD = 5
    qtree = scipy.spatial.KDTree(point_collection)
    
    filtered_point = []
    print "Total #point is: ", len(point_collection)
    count = 0
    for pt in point_collection:
        count += 1
        if count % 1000 == 0:
            print "\t Now at ",count
        nearby_point = qtree.query_ball_point(pt, RADIUS)
        if len(nearby_point) >= THRESHOLD:
            filtered_point.append(pt)
    
    fig = plt.figure(figsize=(30,16))
    ax = fig.add_subplot(121, aspect='equal')
    ax.plot([pt[0] for pt in point_collection],
            [pt[1] for pt in point_collection],
            '.',
            color='gray')
    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])
    
    ax = fig.add_subplot(122, aspect='equal')
    ax.plot([pt[0] for pt in filtered_point],
            [pt[1] for pt in filtered_point],
            '.',
            color='blue')
    ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])
    plt.show()

if __name__ == "__main__":
    sys.exit(main())