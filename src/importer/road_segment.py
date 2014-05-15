#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import cPickle

import numpy as np
import matplotlib.pyplot as plt
import const

class RoadSegment:
    def __init__(self, 
                 center, 
                 direction, 
                 norm_dir,
                 half_length, 
                 half_width):
        self.center = center
        self.direction = direction
        self.norm_dir = norm_dir
        self.half_width = half_width
        self.half_length = half_length
    
    def contain_point(self, pt):
        """ Check if a point fall on this road
        """
        vec = pt - self.center
        length_dir = abs(np.dot(vec, self.direction))
        width_dir = abs(np.dot(vec, self.norm_dir))
        if length_dir <= self.half_length and width_dir <= self.half_width:
            return True
        else:
            return False

