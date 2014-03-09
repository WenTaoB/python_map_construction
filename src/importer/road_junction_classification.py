#!/usr/bin/env python
"""
Classification of road junctions
"""

import sys
import cPickle

from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from skimage.feature import peak_local_max, corner_peaks, hog
from skimage.transform import hough_line,probabilistic_hough_line
from skimage.filter import gaussian_filter

import networkx as nx

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from itertools import cycle

import gps_track
import const

def main():
    ground_truth = np.array([2,2,1,5,2,3,3,3,2,6,2,3,6,3,5,6,3,2,3,2,4,6,2,2,3,3,2,1,3,0,3,3,2,3,3,3,5,3,6,2,2,5,3,6,2,3,3,3,6,2,0,2,0,2,5,2,3,2,2,0,4,2,1,0,2,2,2,0,3,5,3,3,6,3,3,3,3,3,0,3,0,2,3,0,0,3,2,0,0,2,2,3,2,2,2,0,3,2,0,1,3,3,2,3,3,3,3,2,3,0,3,3,1,2,3,2,0,0,0,0,0,0,0,0,3,2,3,2,2,3,3,3,3,3,3,2,2])

    simple_ground_truth = []
    for i in np.arange(len(ground_truth)):
        if ground_truth[i] >=4:
            simple_ground_truth.append(4)
        else:
            simple_ground_truth.append(ground_truth[i])
    simple_ground_truth = np.array(simple_ground_truth)

    with open("test_fig/img_data.dat", "rb") as fin:
        img_data = cPickle.load(fin)

    img = img_data[1]

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(121, aspect='equal')
    ax.imshow(img>0, cmap='gray')
    lines = probabilistic_hough_line(img,line_length=20)
    print len(lines)
    N_BIN = 32
    theta_bins = np.arange(N_BIN)*np.pi/N_BIN
    bin_hist = np.zeros(N_BIN)

    for line in lines:
        ax.plot([line[0][0],line[1][0]],
                [line[0][1],line[1][1]],'-r')
        vec = np.array([line[1][0]-line[0][0], line[1][1]-line[0][1]])*1.0
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1.0:
            vec /= vec_norm

        cross_product = abs(np.dot(vec, np.array([1,0])))
        theta_bin_idx = int(np.arccos(cross_product) / np.pi * N_BIN)
        bin_hist[theta_bin_idx] += 1

    ax.set_xlim([0, img.shape[0]])
    ax.set_ylim([img.shape[1], 0])

    ax = fig.add_subplot(122)
    x_vals = np.arange(N_BIN)*90.0/N_BIN
    ax.plot(x_vals, bin_hist, '.-')

    plt.show()

    #hog_features = []
    #hog_imgs = []
    #for new_img in img_data:
    #    hog_array, hog_image = hog(np.array(new_img>0), visualise=True)
    #    hog_features.append(hog_array)
    #    hog_imgs.append(hog_image)
    #with open("test_fig/hog_features.dat", 'wb') as fout:
    #    cPickle.dump(hog_features, fout, protocol=2)
    #with open("test_fig/hog_imgs.dat", 'wb') as fout:
    #    cPickle.dump(hog_imgs, fout, protocol=2)

    #return

    #with open("test_fig/hog_features.dat", "rb") as fin:
    #    hog_features = cPickle.load(fin)
    #with open("test_fig/hog_imgs.dat", "rb") as fin:
    #    hog_imgs = cPickle.load(fin)

    #n_training = 30
    #hog_features = np.array(hog_features)
    #X = np.array(hog_features[0:n_training,:])
    #Y = simple_ground_truth[0:n_training]
    #clf = tree.DecisionTreeClassifier()
    #clf = clf.fit(X, Y)
    #test_X = hog_features[n_training:-1,:]
    ##prediction = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, Y).predict(test_X)
    #prediction = clf.predict(X)

    #n_correct = 0
    #for i in np.arange(len(prediction)):
    #    #if prediction[i] == simple_ground_truth[n_training+i]:
    #    if prediction[i] == simple_ground_truth[i]:
    #        n_correct += 1.0
    #score = n_correct / len(prediction) * 100
    #print "Correctness: %.2f%%"%score

    #fig = plt.figure(figsize=const.figsize)
    #ax = fig.add_subplot(111)
    ##ax.plot(np.arange(len(prediction)), simple_ground_truth[n_training:-1], 'b.')
    #ax.plot(np.arange(len(prediction)), simple_ground_truth[0:n_training], 'b.')
    #ax.plot(np.arange(len(prediction)), prediction, 'rx')
    #plt.show()

if __name__ == "__main__":
    sys.exit(main())
