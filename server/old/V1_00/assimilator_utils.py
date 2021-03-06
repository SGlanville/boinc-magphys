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
Some utilities for the Assimilator
"""


def is_gzip(outFile):
    """
    Test if the file is a gzip file by opening it and reading the magic number

    >>> is_gzip('/Users/kevinvinsen/Downloads/boinc-magphys/NGC1209__wu719/NGC1209__wu719_0_0.gzip')
    True
    >>> is_gzip('/Users/kevinvinsen/Downloads/boinc-magphys/NGC1209__wu719/NGC1209__wu719_1_0.gzip')
    True
    >>> is_gzip('/Users/kevinvinsen/Downloads/boinc-magphys/NGC1209__wu719/NGC1209__wu719_0_0')
    False
    >>> is_gzip('/Users/kevinvinsen/Downloads/boinc-magphys/NGC1209__wu719/NGC1209__wu719_1_0')
    False
    >>> is_gzip('/Users/kevinvinsen/Downloads/boinc-magphys/NGC1209__wu719/empty')
    False
    """
    result = False
    f = open(outFile, "rb")
    try:
        magic = f.read(2)
        if len(magic) == 2:
            method = ord(f.read(1))

            result = magic == '\037\213' and method == 8

    except IOError:
        pass
    finally:
        f.close()
    return result
