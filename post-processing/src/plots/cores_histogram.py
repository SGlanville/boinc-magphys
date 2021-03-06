#! /usr/bin/env python2.7
#
#    (c) UWA, The University of Western Australia
#    M468/35 Stirling Hwy
#    Perth WA 6009
#    Australia
#
#    Copyright by UWA, 2012-2013
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
"""
Build the os stacks
"""
import logging
import argparse
from plots.usage_mod import plot_cores

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:' + logging.BASIC_FORMAT)

parser = argparse.ArgumentParser('Plot graphs of OS from theSkyNet POGS data ')
parser.add_argument('file', nargs='*', help='the file to plot the OS data to')
args = vars(parser.parse_args())

if len(args['file']) != 1:
    parser.print_help()
    exit(1)


CORES = [
    [4, 6454],
    [2, 5108],
    [8, 3581],
    [1, 1008],
    [6, 490],
    [12, 392],
    [16, 156],
    [3, 108],
    [24, 66],
    [32, 35],
    [256, 20],
    [5, 17],
    [64, 15],
    [7, 8],
    [10, 7],
    [48, 6],
    [9, 5],
    [100, 4],
    [40, 3],
    [128, 3],
    [22, 2],
    [14, 1],
    [18, 1],
    [20, 1],
    [21, 1],
    [50, 1],
    [4048, 1],
    ]

map_of_os = {}

plot_cores(args['file'][0], CORES)

LOG.info('All Done.')
