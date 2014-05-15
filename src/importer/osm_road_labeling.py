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

import const

def main():
    parser = OptionParser()
    parser.add_option("-i","--input_img", dest="input_img", help="Input image filename.", type="string", metavar="InputImage")
    parser.add_option("--test_case", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)

    (options, args) = parser.parse_args()

    if not options.input_img:
        parser.error("Input image file not found!")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    img = cv2.imread(options.input_img, 0)
    img = 255 - img

    sift = cv2.SIFT()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(img, kp, img)
    
    #img = 1 - img

    #img = skimage.morphology.remove_small_objects(img, min_size=1000)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(img, cmap='gray')
    plt.show()
            
if __name__ == "__main__":
    sys.exit(main())
