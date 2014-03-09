#!/usr/bin/env python
"""
Using a unified grid to rasterize GPS tracks.
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

from itertools import cycle

import gps_track
import const

def ij_to_id(node_ij, matrix_shape):
    """ Transform (i,j) to a single integer for node indexing
        Formula: node_id = node[1] + node[0]*matrix_shape[1]
        Args:
            matrix_shape: (n_row, n_col) of the matrix covering the region;
            node_ij: (i,j) of the node index 
        Return:
            node_id: an integer
    """
    return node_ij[1] + node_ij[0]*matrix_shape[1]

def id_to_ij(node_id, matrix_shape):
    """ Transform node_id back to (i,j)
        This is reverse of ij_to_id
        Formula:
            i = node_id / matrix_shape[1]
            j = node_id % matrix_shape[1]
        Args:
            matrix_shape: (n_row, n_col) of the matrix covering the region;
            node_id: single integer representing the node_id according to the formular in ij_to_nodeid
        Return:
            (i,j) representing the node in the array
    """
    i = int(node_id / matrix_shape[1])
    j = node_id % matrix_shape[1]
    return (i,j)

def rasterize_tracks(tracks,
                     bound_box,
                     matrix_shape):
    """
        This function rasterize GPS tracks using a unified grid of matrix_shape around the 
        bound_box.

        Args:
            - tracks: a list of GPS tracks;
            - bound_box: [southeast, northwest];
            - matrix_shape: [n_row, n_col].
        Return:
            - rasterized_tracks: a list of list, [[box_id, ...], [box_id, ...]];
            - point_count_array: a sparse array, point counts for the non-empty boxes;
            - track_indexing_hash: a hash of hash, each recording the tracks touching the grid
                                   box.
    """
    pt_count_hash = {}
    track_indexing_hash = {}
    rasterized_tracks = []

    delta_easting = (bound_box[1][0] - bound_box[0][0]) / matrix_shape[1]
    delta_northing = (bound_box[1][1] - bound_box[0][1]) / matrix_shape[0]
    # Iterate over all GPS point
    track_idx = 0
    for track in tracks:
        rastered_track = []
        for pt in track.utm:
            point_i = int((bound_box[1][1] - pt[1]) / delta_northing)
            point_j = int((pt[0] - bound_box[0][0]) / delta_northing)
            if point_i < 0 or point_i >= matrix_shape[1]-1 or point_j < 0 or\
                    point_j >= matrix_shape[0]-1:
                continue
            rastered_track.append((point_i, point_j))
            if pt_count_hash.has_key((point_i, point_j)):
                pt_count_hash[(point_i, point_j)] += 1.0
                track_indexing_hash[(point_i, point_j)][track_idx] = 1
            else:
                pt_count_hash[(point_i, point_j)] = 1.0
                track_indexing_hash[(point_i, point_j)] = {}
                track_indexing_hash[(point_i, point_j)][track_idx] = 1
        track_idx += 1
        rasterized_tracks.append(rastered_track)

    IJ = np.array(pt_count_hash.keys())
    V = np.array([pt_count_hash[(ij[0],ij[1])] for ij in IJ])
     
    point_count_array = sparse.coo_matrix((V, IJ.T), shape=matrix_shape)
    return rasterized_tracks, point_count_array, track_indexing_hash

def neighbor_search(location,
                    rasterized_tracks,
                    track_indexing_hash,
                    neighborbood,
                    search_depth):
    """
        Args:
            - location: (i,j) of the matrix;
            - rasterized_tracks
            - track_indexing_hash
            - neighborhood: size of the neighborhood. E.g., 1 means 1-neighborhood;
            - search_depth: depths
        Return:
            - neighbors: a list of (i,j)'s.
    """
    neighbors = []
    tracks_idxs = track_indexing_hash[location].keys()
    for idx in tracks_idxs:
        r_track = rasterized_tracks[idx]
        prev = (-1,-1)
        prev_found = False
        nxt = (-1,-1)
        nxt_found = False
        count = 0
        for loc in r_track:
            if loc[0] == location[0] and loc[1] == location[1]:
                if not prev_found and count != 0:
                    prev = r_track[count-1]
                    prev_found = True
                if count < len(r_track) - 1:
                    nxt = r_track[count+1]
                    nxt_found = True
                else:
                    nxt_found = False
            count += 1
        if prev_found:
            neighbors.append(prev)
        if nxt_found:
            neighbors.append(nxt)
    
    if search_depth == 1:
        return neighbors
    else:
        results = []
        for loc in neighbors:
            results.extend(neighbor_search(loc,
                                           rasterized_tracks,
                                           track_indexing_hash,
                                           neighborbood,
                                           search_depth-1))
        return results

def line_segment_intersection(line1,
                              line2):
    """
        Calculate the intersection between two line segments (if they do intersects).
    """
    a = float(line1[0][0]*line1[1][1] - line1[0][1]*line1[1][0])
    b = float(line1[0][1] - line1[1][1])
    c = float(line1[1][0] - line1[0][0])

    d = float(line2[0][0]*line2[1][1] - line2[0][1]*line2[1][0])
    e = float(line2[0][1] - line2[1][1])
    f = float(line2[1][0] - line2[0][0])

    prod = b*f - c*e
    if abs(prod) < 1e-10:
        return (np.inf, np.inf)

    xc = (d*c - a*f) / prod
    yc = (a*e - b*d) / prod

    sign_x1 = (xc - line1[0][0])*(xc - line1[1][0])
    sign_y1 = (yc - line1[0][1])*(yc - line1[1][1])

    if sign_x1 > 1e-10:
        return (np.inf, np.inf)
    if sign_x1 < 1e-10:
        if sign_y1 > 1e-10:
            return (np.inf, np.inf)

    sign_x2 = (xc - line2[0][0])*(xc - line2[1][0])
    sign_y2 = (yc - line2[0][1])*(yc - line2[1][1])

    if sign_x2 > 1e-10:
        return (np.inf, np.inf)
    if sign_x2 == 1e-10:
        if sign_y2 > 1e-10:
            return (np.inf, np.inf)
    return (int(xc), int(yc))

def main():
    tracks = gps_track.load_tracks(sys.argv[1])

    # Write 3D for Yangyan: skeleton
    #delta_easting = const.RANGE_NE[0] - const.RANGE_SW[0]
    #delta_northing = const.RANGE_NE[1] - const.RANGE_SW[1]
    #f = open("test_region_3D.txt", "w")
    #for track in tracks:
    #    for pt in track.utm:
    #        #if pt[0]<=const.RANGE_SW[0]+3000 and pt[0]>=const.RANGE_SW[0]+2000:
    #        #    if pt[1]<=const.RANGE_SW[1]+3000 and pt[1]>=const.RANGE_SW[1]+2000:
    #        pe = (pt[0] - const.RANGE_SW[0]) / delta_easting * 10
    #        pn = (pt[1] - const.RANGE_SW[1]) / delta_northing * 10
    #        f.write("%.6f %.6f %.6f\n"%(pe,pn,0.01*np.random.rand()))

    #f.close()
    #return

    N_ROW = 1000 # Divide northing
    N_COL = 1000 # Divide southing

    """
        test case index:
            0: Beijing
            1: SF small
    """
    test_case = 0

    if test_case == 0:
        rasterized_tracks, point_count_array, track_indexing_hash =\
                                    rasterize_tracks(tracks,
                                                     [const.RANGE_SW, const.RANGE_NE],
                                                     (N_ROW, N_COL)) 

    else:
        rasterized_tracks, point_count_array, track_indexing_hash =\
                                        rasterize_tracks(tracks,
                                                     [const.SF_small_RANGE_SW, const.SF_small_RANGE_NE],
                                                     (N_ROW, N_COL)) 


    dense_array = np.array(point_count_array.todense())

    lines = probabilistic_hough_line(dense_array,line_length=50)
    print len(lines)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    
    ax.imshow(dense_array>0, cmap='gray')

    for line in lines:
        ax.plot([line[0][0],line[1][0]],
                [line[0][1],line[1][1]],'-r')
    ax.set_xlim([0, N_ROW])
    ax.set_ylim([N_COL, 0])
    plt.show()
    return

    intersections = []
    angle_threshold = 0.71
    for line_i in range(0, len(lines)):
        for line_j in range(line_i+1, len(lines)):
            line1 = lines[line_i]
            line2 = lines[line_j]
            vec1 = 1.0*np.array([line1[1][0]-line1[0][0], line1[1][1]-line1[0][1]]) 
            vec2 = 1.0*np.array([line2[1][0]-line2[0][0], line2[1][1]-line2[0][1]]) 
            
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)

            if abs(np.dot(vec1, vec2)) < angle_threshold:
                pc = line_segment_intersection(line1, line2)
                if pc[0] != np.inf:
                    intersections.append(pc)

    new_img = np.zeros((N_ROW, N_COL))

    for itr in intersections:
        new_img[itr[0],itr[1]] += 1
    
    peaks = corner_peaks(new_img, min_distance=20)

    ax = fig.add_subplot(122, aspect='equal')
    ax.imshow(dense_array>0, cmap='gray')
    for peak in peaks:
        ax.plot(peak[0], peak[1], '.r', markersize=12)
    ax.set_xlim([0, N_ROW])
    ax.set_ylim([N_COL, 0])
    plt.show()
    return


    window_size = 40
   
    i = 0

    img_data = []
    hog_features = []
    hog_imgs = []
    for peak in peaks:
        if peak[0]-window_size<0 or peak[0]+window_size>N_ROW or\
                peak[1]-window_size<0 or peak[1]+window_size>N_COL:
            continue
            
        fig = plt.figure(figsize=const.figsize)
        ax = fig.add_subplot(111, aspect='equal')
        
        ax.imshow(dense_array>0, cmap='gray')
        ax.plot(peak[0], peak[1], 'rx', markersize=12)

        ax.set_xlim([peak[0]-window_size, peak[0]+window_size])
        ax.set_ylim([peak[1]+window_size, peak[1]-window_size])

        plt.savefig("test_fig/fig_%d.png"%i)
        plt.close()

        new_img = dense_array[(peak[1]-window_size):(peak[1]+window_size), \
                              (peak[0]-window_size):(peak[0]+window_size)]
        img_data.append(new_img) 
        hog_array, hog_image = hog(np.array(new_img>0), pixels_per_cell=(4, 4), visualise=True)
        hog_features.append(hog_array)
        hog_imgs.append(hog_image)

        i += 1

    with open("test_fig/img_data.dat", "wb") as fout:
        cPickle.dump(img_data, fout, protocol=2 )
    with open("test_fig/hog_features.dat", 'wb') as fout:
        cPickle.dump(hog_features, fout, protocol=2)
    with open("test_fig/hog_imgs.dat", 'wb') as fout:
        cPickle.dump(hog_imgs, fout, protocol=2)

    return
    colors = cycle('bgrcmybgrcmybgrcmykbgrcmy')

    #ax.imshow(dense_array>0, cmap='gray')


    window_size = 80
    chosen_ij = (752, 814)
    #chosen_ij = (370, 408)
    search_range = 4
    G = nx.Graph()

    active_track_idxs = {}
    for i in range(chosen_ij[1]-window_size, chosen_ij[1]+window_size):
        for j in range(chosen_ij[0]-window_size, chosen_ij[0]+window_size):
            if dense_array[i,j] > 0:
                # Add graph nodes
                G.add_node((i,j))
                # Add edge to nearby nodes
                for k in range(i-search_range, i+search_range+1):
                    if k < chosen_ij[1]-window_size or k >= chosen_ij[1]+window_size:
                        continue
                    for l in range(j-search_range, j+search_range+1):
                        if l < chosen_ij[0]-window_size or l >= chosen_ij[0]+window_size:
                            continue
                        if dense_array[k,l] > 0:
                            G.add_edge((i,j),(k,l),{'w':1.0})

                for idx in track_indexing_hash[(i,j)].keys():
                    active_track_idxs[idx] = 1
  
    ax.set_xlim([chosen_ij[0]-window_size, chosen_ij[0]+window_size])
    ax.set_ylim([chosen_ij[1]-window_size, chosen_ij[1]+window_size])

    # Iterate over active tracks to add constraints
    to_break = False
    count = 0
    preserved_edges = {}
    for idx in active_track_idxs.keys():
        # Iterate over it's nodes
        for loc_idx in range(0, len(rasterized_tracks[idx])-1):
            cur_loc = rasterized_tracks[idx][loc_idx]
            nxt_loc = rasterized_tracks[idx][loc_idx+1]

            if abs(cur_loc[0]-nxt_loc[0])+abs(cur_loc[1]-nxt_loc[1])<2*search_range:
                continue
            cur_loc_ok = False
            nxt_loc_ok = False
            
            if cur_loc[0]>=chosen_ij[1]-window_size and cur_loc[0]<chosen_ij[1]+window_size:
                if cur_loc[1]>=chosen_ij[0]-window_size and\
                        cur_loc[1]<chosen_ij[0]+window_size:
                    cur_loc_ok = True
            if nxt_loc[0]>=chosen_ij[1]-window_size and nxt_loc[0]<chosen_ij[1]+window_size:
                if nxt_loc[1]>=chosen_ij[0]-window_size and\
                        nxt_loc[1]<chosen_ij[0]+window_size:
                    nxt_loc_ok = True
            
            if cur_loc_ok and nxt_loc_ok: 
                can_be_connected = nx.algorithms.has_path(G, cur_loc, nxt_loc)
                if can_be_connected:
                    count += 1
                    path = nx.shortest_path(G, source=cur_loc, target=nxt_loc, weight='w')
                    # Record edge
                    for node_idx in range(0,len(path)-1):
                        edge = (path[node_idx],path[node_idx+1])
                        preserved_edges[edge] = 1
                        # Reduce edge weight
                        G[path[node_idx]][path[node_idx+1]]['w'] /= 1.05 
                    #ax.plot([cur_loc[1],nxt_loc[1]],
                    #        [cur_loc[0],nxt_loc[0]],
                    #        'rx-')
        if to_break:
            break
    print "Count = ",count

    for edge in preserved_edges.keys():
        ax.plot([edge[0][1],edge[1][1]],
                [edge[0][0],edge[1][0]],'-r')

    plt.show()
    return
    chosen_i = np.random.randint(window_size, N_ROW-window_size)
    chosen_j = np.random.randint(window_size, N_COL-window_size)
    test_image = np.array(dense_array[chosen_ij[1]-window_size:chosen_ij[1]+window_size,\
                             chosen_ij[0]-window_size:chosen_ij[0]+window_size] > 0)

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(test_image>0, cmap='gray')

    plt.show()
    
if __name__ == "__main__":
    sys.exit(main())
