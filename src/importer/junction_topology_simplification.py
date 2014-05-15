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

import osm_parser
import const

def main():
    usage = "usage: %prog [options] junction.osm"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    
    if len(args) != 1:
        parser.error("Incorrect number of arguments")

    G = osm_parser.read_osm(args[0])

    fig = plt.figure(figsize=const.figsize)
    ax = fig.add_subplot(111, aspect='equal')
    osm_parser.draw_osm(G, ax)
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
