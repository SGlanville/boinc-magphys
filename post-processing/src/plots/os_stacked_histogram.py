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
from plots.usage_mod import plot_os_data_stack

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:' + logging.BASIC_FORMAT)

parser = argparse.ArgumentParser('Plot graphs of OS from theSkyNet POGS data ')
parser.add_argument('file', nargs='*', help='the file to plot the OS data to')
args = vars(parser.parse_args())

if len(args['file']) != 1:
    parser.print_help()
    exit(1)


DATA = [
    ['Microsoft Windows 7', 5386],
    ['Android', 4029],
    ['Linux', 1817],
    ['Microsoft Windows XP', 1702],
    ['Microsoft Windows 8', 873],
    ['Microsoft Windows Vista', 494],
    ['Darwin', 460],
    ['Microsoft Windows Server 2008 "R2"', 288],
    ['Microsoft Windows Server 2003', 70],
    ['Microsoft Windows Server 2003 "R2"', 41],
    ['Microsoft Windows Server 2008', 28],
    ['Unknown', 19],
    ['Microsoft Windows 2000', 19],
    ['Microsoft Windows Server 2012', 12],
    ['FreeBSD', 11],
    ['Microsoft Windows 8 Server', 10],
    ['Microsoft Windows 8.1', 10],
    ['Microsoft Windows Server "Longhorn"', 9],
    ['Microsoft', 9],
    ['Microsoft Windows XP Professional x64 Edition', 5],
    ['SunOS', 4],
    ['Microsoft Windows 98', 4],
    ['OpenBSD', 2],
    ['Microsoft Windows Millennium', 1],
    ]

GROUPS = {
    'Microsoft': ['Microsoft'],
    'Android': ['Android'],
    'Linux': ['Linux'],
    'BSD': ['Darwin', 'FreeBSD', 'OpenBSD'],
    'Unknown': ['Unknown'],
    '*Nix': ['SunOS'],
}

map_of_os = {}

for item in DATA:
    name = item[0]
    count = item[1]

    group_name = None
    for group, values in GROUPS.iteritems():
        for group_element in values:
            if name.startswith(group_element):
                group_name = group
                break

        if group_name is not None:
            break

    if group_name is None:
        group_name = name

    list_of_items = map_of_os.get(group_name)
    if list_of_items is None:
        list_of_items = []
        map_of_os[group_name] = list_of_items

    list_of_items.append([name, count])

plot_os_data_stack(args['file'][0], map_of_os)

LOG.info('All Done.')