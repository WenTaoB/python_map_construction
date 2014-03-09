#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import gps_track
import const

def dp_segmentation_idx(track, start_idx, end_idx, d_threshold):
    if start_idx >= end_idx-1:
        return []

    pt_vec = []
    for i in range(start_idx, end_idx):
        pt_vec.append((track.utm[i][0], track.utm[i][1]))
    pt_vec = np.array(pt_vec)

    vec_start_end = pt_vec[-1] - pt_vec[0]
    vec_length = np.linalg.norm(vec_start_end)
    if vec_length < 0.01:
        return []
    vec_start_end /= vec_length
    vec_norm = np.array((-1*vec_start_end[1], vec_start_end[0]))

    delta_pt = pt_vec - pt_vec[0]

    dist_to_line = np.abs(np.dot(delta_pt, vec_norm))
    max_dist_idx = np.argmax(dist_to_line)
    max_pt_idx = max_dist_idx + start_idx

    if dist_to_line[max_dist_idx] > d_threshold:
        part1 = dp_segmentation_idx(track, start_idx, max_pt_idx+1, d_threshold)
        part2 = dp_segmentation_idx(track, max_pt_idx, end_idx, d_threshold)

        result = []
        result.extend(list(part1))
        result.extend(list(part2))
        return result

    return [start_idx, end_idx]

def dp_segmentation(tracks, d_threshold):
    segments = []
    seg_from = []
    track_idx = 0
    for track in tracks:
        seg_idx = dp_segmentation_idx(track, 0, len(track.utm), d_threshold)
        for i in range(0, len(seg_idx)-1):
            segment = []
            if seg_idx[i] >= seg_idx[i+1]:
                continue
            for j in range(seg_idx[i], seg_idx[i+1]):
                segment.append((track.utm[j][0], track.utm[j][1]))
            if len(segment) > 2:
                segments.append(np.array(segment))
                seg_from.append(track_idx)
        
        track_idx += 1

    return segments, seg_from

def main():
    tracks = gps_track.load_tracks(sys.argv[1])

    track = tracks[2]
    segments, seg_from = dp_segmentation([track], 5)
    print "There are %d segments."%(len(segments))

    color_strings = ['b', 'r', 'c', 'm', 'y', 'g']
    colors = []
    for i in range(0, len(segments)):
        colors.append(color_strings[i%6])

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, aspect='equal')
    ax.plot([pt[0] for pt in track.utm],
            [pt[1] for pt in track.utm],
            'k+-')
    collection = LineCollection(segments, colors=colors, linewidth=3)
    ax.add_collection(collection)
    #ax.set_xlim([const.SF_small_RANGE_SW[0], const.SF_small_RANGE_NE[0]])
    #ax.set_ylim([const.SF_small_RANGE_SW[1], const.SF_small_RANGE_NE[1]])
    #ax.set_xlim([const.RANGE_SW[0], const.RANGE_NE[0]])
    #ax.set_ylim([const.RANGE_SW[1], const.RANGE_NE[1]])

    plt.show()

if __name__ == "__main__":
    sys.exit(main())
