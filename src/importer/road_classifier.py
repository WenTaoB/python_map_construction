#!/usr/bin/env python

"""
Created on Wed Oct 09 16:56:33 2013
@author: ChenChen
"""
import sys
import cPickle
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import svm, preprocessing

from point_cloud import PointCloud
import const

def compute_feature(point_cloud,
                    img,
                    line_img,
                    N_bins,
                    feature_window,
                    LOC,
                    R):

    """
    """
    print "Computing features..."
    point_cloud_bins = np.zeros((N_bins, N_bins))
    delta_x = 2 * R / N_bins
    delta_y = 2 * R / N_bins
    for i in np.arange(point_cloud.directions.shape[0]):
        x = point_cloud.locations[i,0] - LOC[0] + R
        y = LOC[1] + R - point_cloud.locations[i, 1]
        i_x = int(x / delta_x)
        i_y = int(y / delta_y)
        if i_x >= 0 and i_x < N_bins and i_y >=0 and i_y < N_bins:
            point_cloud_bins[i_y, i_x] += 1
    
    feature_length = (2*feature_window)**2
    features = []
    feature_locations = []
    for i in range(feature_window, N_bins-feature_window):
        for j in range(feature_window, N_bins-feature_window):
            feature_matrix = np.copy(point_cloud_bins[i-feature_window:i+feature_window, j-feature_window:j+feature_window])
            sum_val = sum(sum(feature_matrix))
            if sum_val > 0:
                feature_matrix /= sum_val

            line_feature_matrix = np.copy(point_cloud_bins[i-feature_window:i+feature_window, j-feature_window:j+feature_window])
            feature_matrix = np.append(feature_matrix, line_feature_matrix, 0)
            reshaped_feature = np.reshape(feature_matrix, feature_matrix.shape[0]*feature_matrix.shape[1])
            features.append(reshaped_feature)
            feature_locations.append((i,j))
    orig_features = np.array(features)

    # Feature scaling
    features = preprocessing.scale(orig_features)

    print "Computing labels..."
    labels = []
    for idx in np.arange(len(feature_locations)):
        orig_i = feature_locations[idx][0]*10
        orig_j = feature_locations[idx][1]*10
        delta = 3
        value_matrix = img[orig_i-delta:orig_i+delta+1, orig_j-delta:orig_j+delta+1]
        if sum(sum(value_matrix)) > 255:
            if point_cloud_bins[feature_locations[idx][0], feature_locations[idx][1]] > 0:
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(0)

    labels = np.array(labels)

    return features, feature_locations, labels, point_cloud_bins

def main():
    parser = OptionParser()
    parser.add_option("-i","--input_img", dest="input_img", help="Input image filename.", type="string", metavar="InputImage")
    parser.add_option("--point_cloud1", dest="input_pointcloud1", help="Input pointcloud filename.", type="string", metavar="InputPointCloud")
    parser.add_option("--point_cloud_line1", dest="input_pointcloud_line1", help="Input pointcloud filename.", type="string", metavar="InputPointCloud")
    parser.add_option("--point_cloud2", dest="input_pointcloud2", help="Input pointcloud filename.", type="string", metavar="InputPointCloud")
    parser.add_option("--point_cloud_line2", dest="input_pointcloud_line2", help="Input pointcloud filename.", type="string", metavar="InputPointCloud")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)

    (options, args) = parser.parse_args()

    if not options.input_img:
        parser.error("Input image file not found!")
    if not options.input_pointcloud1:
        parser.error("Input pointcloud1 file not found!")
    if not options.input_pointcloud2:
        parser.error("Input pointcloud2 file not found!")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    with open(options.input_pointcloud1, 'rb') as fin:
        point_cloud1 = cPickle.load(fin)

    with open(options.input_pointcloud2, 'rb') as fin:
        point_cloud2 = cPickle.load(fin)

    img = cv2.imread(options.input_img, 0)
    img = 255 - img

    # Compute features
    feature_window = 4 #result will be 9x9 window 
    N_bins = 100
    line_img = cv2.imread(options.input_pointcloud_line1, 0)
    line_img = 255 - line_img
    features, feature_locations, labels, point_cloud_bins = \
                                          compute_feature(point_cloud1,
                                                          img,
                                                          line_img,
                                                          N_bins,
                                                          feature_window,
                                                          LOC,
                                                          R)
    

    LOC1 = const.Region_1_LOC
    img1 = cv2.imread("test_region_1.png", 0)
    img1 = 255 - img1

    line_img1 = cv2.imread(options.input_pointcloud_line2, 0)
    line_img1 = 255 - line_img1
    #features, feature_locations, labels, point_cloud_bins = \
    #                                      compute_feature(point_cloud2,
    #                                                      img1,
    #                                                      line_img1,
    #                                                      N_bins,
    #                                                      feature_window,
    #                                                      LOC1,
    #                                                      R)
    print "Start training"
    clf = svm.SVC()
    #weight = labels + 0.1
    num_to_train = int(0.5*len(labels))
    clf.fit(features[:num_to_train], labels[:num_to_train])
    #clf.fit(features0, labels0)

    new_img = np.array(255*(point_cloud_bins > 1), np.uint8)

    #min_line_length = 5 
    #max_line_gap = 5
    #lines = cv2.HoughLinesP(new_img, 1, np.pi/180, 10, min_line_length, max_line_gap)
    
    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(121, aspect='equal')
    ax.imshow(img, cmap='gray')
    ax.set_xlim([0, 1000])
    ax.set_ylim([1000, 0])

    ax = fig.add_subplot(122, aspect='equal')
    ax.imshow(new_img, cmap='gray')
    
    #for line in lines:
    #    x1,y1,x2,y2 = line[0]
    #    ax.plot([x1,x2], [y1,y2], 'r-')

    prediction = clf.predict(features)
    correct_count = 0
    for idx in range(0, len(prediction)):
        #if labels[idx] == 1:
        #    ax.plot(feature_locations[idx][1], feature_locations[idx][0], '.b')
            
        if labels[idx] == prediction[idx]:
            if labels[idx] == prediction[idx] and labels[idx] == 1:
                correct_count += 1
                ax.plot(feature_locations[idx][1], feature_locations[idx][0], 'ob')
                continue

        if labels[idx] != prediction[idx]:
            if prediction[idx] == 1:
                ax.plot(feature_locations[idx][1], feature_locations[idx][0], '.g')
            else:
                ax.plot(feature_locations[idx][1], feature_locations[idx][0], '.r')

    correctness = 100*float(correct_count)/sum(labels)
    print "Correctess: %.2f%%"%(correctness)
    ax.set_xlim([0, N_bins])
    ax.set_ylim([N_bins, 0])
    
    plt.show()
            
if __name__ == "__main__":
    sys.exit(main())
