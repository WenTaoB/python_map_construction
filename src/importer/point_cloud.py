#!/usr/bin/env python
"""
Created on Wed Oct 09 16:56:33 2013

@author: ChenChen
"""

import cPickle

import numpy as np
import matplotlib.pyplot as plt
import const

class PointCloud:
    def __init__(self):
        pass

    def __init__(self, points, directions, track_ids):
        self.locations = np.array(points)
        self.directions = np.array(directions)
        self.track_ids = np.array(track_ids)

    def extract_point_cloud(self, tracks, loc, R):
        locations = []
        directions = []
        track_ids = []
        for track_idx in range(0, len(tracks)):
            track = tracks[track_idx]
            for pt_idx in range(0, len(track.utm)):
                pt = track.utm[pt_idx]
                if pt[0]>=loc[0]-R and pt[0]<=loc[0]+R and \
                        pt[1]>=loc[1]-R and pt[1]<=loc[1]+R:
                    locations.append((pt[0], pt[1]))
                   
                    dir1 = np.array((0.0, 0.0))
                    if pt_idx > 0:
                        dir1 = np.array((track.utm[pt_idx][0]-track.utm[pt_idx-1][0], track.utm[pt_idx][1]-track.utm[pt_idx-1][1]))

                    dir2 = np.array((0.0, 0.0)) 
                    if pt_idx < len(track.utm) - 1:
                        dir2 = np.array((track.utm[pt_idx+1][0]-track.utm[pt_idx][0], track.utm[pt_idx+1][1]-track.utm[pt_idx][1]))

                    direction = dir1 + dir2
                    
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1.0:
                        direction /= direction_norm
                    else:
                        direction *= 0.0
                    
                    directions.append(direction)
                    track_ids.append(track_idx)
        self.locations = np.array(locations)
        self.directions = np.array(directions)
        self.track_ids = np.array(track_ids)

    def visualize_point_cloud(self, loc, R):
        fig = plt.figure(figsize=const.figsize)
        ax = fig.add_subplot(121, aspect="equal")
        ax.plot(self.locations[:,0], self.locations[:,1], '.')
        ax.set_xlim([loc[0]-R, loc[0]+R]) 
        ax.set_ylim([loc[1]-R, loc[1]+R]) 

        ax = fig.add_subplot(122, aspect="equal")
        ax.quiver(self.locations[:,0],
                  self.locations[:,1],
                  self.directions[:,0],
                  self.directions[:,1]) 
        ax.set_xlim([loc[0]-R, loc[0]+R]) 
        ax.set_ylim([loc[1]-R, loc[1]+R]) 
        plt.show()

    def save(self, filename):
        with open(filename, "wb") as fout:
            cPickle.dump(self, fout, protocol=2)
