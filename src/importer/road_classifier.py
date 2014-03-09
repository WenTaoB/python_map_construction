#!/usr/bin/env python
"""
Created on Mon Oct 07 17:19:49 2013

@author: ChenChen
"""
import sys
import cPickle
import math

import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM

import const

def main():
    if len(sys.argv) != 2:
        print "ERROR! Correct usage is:"
        print "\tpython road_classifier.py [gps_point_collection.dat]"
        return
        
    # Load GPS points
    with open(sys.argv[1], "rb") as fin:
        point_collection = cPickle.load(fin)
    GRID_SIZE = 500
    results = np.zeros((GRID_SIZE, GRID_SIZE), np.float)
    for pt in point_collection:
        y_ind = math.floor((pt[0] - const.RANGE_SW[0]) / (const.RANGE_NE[0] -const.RANGE_SW[0]) * GRID_SIZE)
        x_ind = math.floor((pt[1] - const.RANGE_NE[1]) / (const.RANGE_SW[1] -const.RANGE_NE[1]) * GRID_SIZE)
        results[x_ind, y_ind] += 1.0
        if results[x_ind, y_ind] >= 64:
            results[x_ind, y_ind] = 63
    results /= np.amax(results)
    
    thresholded_results = np.zeros((GRID_SIZE, GRID_SIZE), np.bool)    
    THRESHOLD = 0.02
    for i in range(0, GRID_SIZE):
        for j in range(0, GRID_SIZE):
            if results[i,j] >= THRESHOLD:
                thresholded_results[i,j] = 1
            else:
                thresholded_results[i,j] = 0
    
    training_label = [0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0]
    
    eval_label = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0]    
    
    with open("training_feature_collection.dat", "rb") as fin:
        training_feature = cPickle.load(fin)
    with open("training_feature_loc.dat", "rb") as fin:
        training_feature_loc = cPickle.load(fin)
        
    with open("eval_feature_collection.dat", "rb") as fin:
        eval_feature = cPickle.load(fin)
    with open("eval_feature_loc.dat", "rb") as fin:
        eval_feature_loc = cPickle.load(fin)

    clf = sklearn.svm.SVC(kernel='linear').fit(eval_feature, eval_label)
        
    fig = plt.figure(figsize=(16,16))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    for ind in range(0, len(training_label)):
        if training_label[ind] == 0:
            ax1.plot(range(0, len(training_feature[ind])), training_feature[ind], '.-')
        else:
            ax2.plot(range(0, len(training_feature[ind])), training_feature[ind], '.-')
        if clf.predict(eval_feature[ind]) != eval_label[ind]:
            print "Wrong!, label = ", eval_label[ind]
    
#    ax.imshow(thresholded_results, cmap=CM.gray_r)
#    
#    correct_count = 0
#    one_count = 0
#    for ind in range(0, len(eval_feature)):
#        pred_res = clf.predict(eval_feature[ind])
#        if clf.predict(training_feature[ind]) == 1:
#            print "hello!"
#        #print "delta: ",(clf.predict(np_training_feature[ind]) - training_label[ind])
#        if pred_res == 1:
#            one_count += 1
#        if pred_res == eval_label[ind]:
#            correct_count += 1
#        else:
#            ax.plot(eval_feature_loc[ind][1], eval_feature_loc[ind][0], '.r')
#    
#    print "Prediction precision is %d."%(correct_count)
#    print "There are %d 1s."%(one_count)
#
#    
##    for ind in range(0, len(training_feature)):
##        pred_res = svc.predict(training_feature[ind])
##        if pred_res == 0:
##            ax.plot(training_feature_loc[ind][1], training_feature_loc[ind][0], '.r')
##        elif pred_res == 1:
##            ax.plot(training_feature_loc[ind][1], training_feature_loc[ind][0], '.b')
##        else:
##            print "Error prediction results!"
#    ax.set_xlim([0, GRID_SIZE])
#    ax.set_ylim([0, GRID_SIZE])
    plt.show()
    
        
if __name__ == "__main__":
    sys.exit(main())