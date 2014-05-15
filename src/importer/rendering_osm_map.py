#!/usr/bin/env python

import sys
from optparse import OptionParser

import matplotlib.pyplot as plt

import osm_parser
import const

def main():
    parser = OptionParser()
    parser.add_option("-i", "--osm", dest="osm_filename", help="Input openstreetmap filename", metavar="OSM_FILE", type="string")
    parser.add_option("--testcase", dest="test_case", type="int", help="Test cases: 0: region-0; 1: region-1; 2: SF-region.", default=0)
    (options, args) = parser.parse_args()

    if not options.osm_filename:
        parser.error("Input sample_point_cloud filename not found!")

    R = const.R 
    if options.test_case == 0:
        LOC = const.Region_0_LOC
    elif options.test_case == 1:
        LOC = const.Region_1_LOC
    elif options.test_case == 2:
        LOC = const.SF_LOC
    elif options.test_case == 3:
        sw = const.BJ_TEST_SW
        ne = const.BJ_TEST_NE
        mean_e = 0.5*sw[0] + 0.5*ne[0]
        mean_n = 0.5*sw[1] + 0.5*ne[1]
        LOC = (mean_e, mean_n)
        R = mean_e - sw[0]
    else:
        parser.error("Test case indexed %d not supported!"%options.test_case)

    G = osm_parser.read_osm(options.osm_filename) 
    patches = osm_parser.visualize_osm(G)

    junction_locations = [(39.923614, 116.356207), (39.940455, 116.355552), (39.949844, 116.379493),
                          (39.949122, 116.408263), (39.949440, 116.433447), (39.941108, 116.433774),
                          (39.933702, 116.434112), (39.924491, 116.434512), (39.908591, 116.435759),
                          (39.900882, 116.436435), (39.893719, 116.444111), (39.884558, 116.445109),
                          (39.870635, 116.439439), (39.870887, 116.421436), (39.872231, 116.414709),
                          (39.871980, 116.399388), (39.871584, 116.387812), (39.868085, 116.348549),
                          (39.889551, 116.349094), (39.897579, 116.349166), (39.899081, 116.356591),
                          (39.907176, 116.356575), # 2nd Ring
                          (39.924078, 116.310198), (39.942216, 116.310117), (39.962777, 116.308460),
                          (39.967777, 116.353993), (39.968188, 116.380429), (39.968570, 116.394618),
                          (39.968924, 116.407396), (39.969586, 116.431783), (39.958098, 116.453991),
                          (39.908359, 116.461780), (39.876216, 116.461121), (39.862541, 116.454775),
                          (39.857097, 116.400057), (39.856121, 116.371508), (39.854713, 116.364894),
                          (39.849598, 116.346311), (39.867453, 116.312537), (39.884735, 116.310574),
                          (39.897052, 116.310437), (39.907452, 116.310268), # 3rd Ring
                          (39.897443, 116.274031), (39.924226, 116.275040), (39.946750, 116.274734),
                          (39.969333, 116.275313), (39.974854, 116.284492), (39.984621, 116.299405),
                          (39.986902, 116.353500), (39.987667, 116.378305), (39.988082, 116.393722),
                          (39.988447, 116.407546), (39.987662, 116.442544), (39.971656, 116.469065),
                          (39.922947, 116.489842), (39.915531, 116.489842), (39.907860, 116.489843),
                          (39.871277, 116.489065), (39.842261, 116.478497), (39.832314, 116.423544),
                          (39.832239, 116.401293), (39.831102, 116.346259), (39.831292, 116.290630),
                          (39.847896, 116.283613), (39.865294, 116.283388), (39.875258, 116.278066), #4th
                          (39.894315, 116.211139), (39.924732, 116.212158), (39.941953, 116.211871),
                          (39.955322, 116.216203), (39.992547, 116.222184), (40.013997, 116.287227),
                          (40.020850, 116.327418), (40.023232, 116.353800), (40.023010, 116.385514),
                          (40.022296, 116.417046), (40.020636, 116.441245), (40.016330, 116.455772),
                          (39.999673, 116.499567), (39.958698, 116.529136), (39.941434, 116.541506),
                          (39.909206, 116.544285), (39.868976, 116.548405), (39.845603, 116.543931),
                          (39.833451, 116.527108), (39.819840, 116.505157), (39.814986, 116.492068),
                          (39.805632, 116.475534), (39.786483, 116.430935), (39.762281, 116.401790),
                          (39.778473, 116.342534), (39.780979, 116.317268), (39.777491, 116.295971),
                          (39.845895, 116.225794)]

    R = 400
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    for patch in patches:
        ax.add_patch(patch)

    for node in G.nodes():
        neighbors = G.neighbors(node)
        if len(neighbors) >= 3:
            ax.plot(G.node[node]['data'][0], G.node[node]['data'][1], 'ro')
    #ax.set_xlim([const.BJ_SW[0], const.BJ_NE[0]])
    #ax.set_ylim([const.BJ_SW[1], const.BJ_NE[1]])
    count = 0
    easting_offset = 520
    northing_offset = 145

    for loc in junction_locations:
        print count
        count += 1
        easting, northing = const.BJ_utm_projector(loc[1], loc[0])
        easting -= easting_offset
        northing -= northing_offset
        ax.set_title("junction-%d"%count)
        ax.set_xlim([easting-R, easting+R])
        ax.set_ylim([northing-R, northing+R])
        
        fig.savefig("junction_images/%d.png"%count, dpi=100)

    plt.close()

if __name__ == "__main__":
    sys.exit(main())
