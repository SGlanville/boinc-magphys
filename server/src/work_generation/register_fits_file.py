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
Register a FITS file ready to be converted into Work Units
"""
import argparse
import sys
from datetime import datetime
from sqlalchemy import select
from utils.logging_helper import config_logger
import os
from sqlalchemy.engine import create_engine
from config import DB_LOGIN
from database.database_support_core import REGISTER, TAG_REGISTER, TAG
import re

LOG = config_logger(__name__)
LOG.info('PYTHONPATH = {0}'.format(sys.path))

parser.add_argument('filename', type=string, nargs=1, help='The file name')
parser.add_argument('directory', type=string, nargs=1, help='The directory')

parser.add_argument('priority', type=int, nargs=1, help='the higher the number the higher the priority')
parser.add_argument('run_id', type=int, nargs=1, help='the run id to be used')
parser.add_argument('tags', nargs='*', help='any tags to be associated with the galaxy')

PRIORITY = args['priority'][0]
RUN_ID = args['run_id'][0]
TAGS = args['tags']

path = "/home/ec2-user/galaxies_TAR/" #path where the tar files are kept before registration
final_path ="/home/ecu2-user/galaxies/"
galaxy_direct = os.listdir(path) #creates a list of the tar file names 

# Make sure the file exists
if not os.path.isfile(INPUT_FILE):
    LOG.error('The file %s does not exist', INPUT_FILE)
    exit(1)

# Connect to the database - the login string is set in the database package
ENGINE = create_engine(DB_LOGIN)
connection = ENGINE.connect()
transaction = connection.begin()

galaxies_to_register = []

class GalaxyToRegister:
    def __init__(self, galaxy_name, redshift, type):
        self.galaxy_name=galaxy_name
        self.redshift=redshift
        self.type=type
    
    def sanitize_galaxy_name(self):
        """ Return a sanitized version of the galaxy_name
        >>>sanitize_galazy_name(EBHH000??H)
        EBHH000H
        """
        bad_characters = re.compile('[()[\]{}&*#$:<>?~%"\\|/]') 
        re.sub(bad_characters, '', self.galaxy_name)   
    
    def correct_redshift(self):
        """If redshift value = 0.05, take 0.0001 off it"""
        correct_redshift = self.redshift;
        if self.redshift == 0.005:
           correct_redshift = 0.05-0.001
           return correct_redshift          
        
        
        
#Loop through the file, reading each line
f = open('INPUT_FILE', 'r')
f.readline()
for line in f: 
    initial_list = line.split(' ', 4)
    line_list = [initial_list[0], initial_list[3], initial_list[4]]
    galaxy_line = GalaxyToRegister(initial_list[0], initial_list[3], initial_list[4])
    galaxies_to_register.extend(galaxy_line)
    
f.close()
    
#sort galaxies to register alphabetically 
sorted(galaxies_to_register, key = lambda galaxy: galaxy_to_register.galaxy_name)

    
# If it is a float store it as the sigma otherwise assume it is a string pointing to a file containing the sigmas
try:
    sigma = float(SIGMA)
    sigma_filename = None
except ValueError:
    sigma = 0.0
    sigma_filename = SIGMA


register_id = result.inserted_primary_key[0]

# Get the tag ids
tag_ids = set()
for tag_text in TAGS:
    tag_text = tag_text.strip()
    if len(tag_text) > 0:
        tag = connection.execute(select([TAG]).where(TAG.c.tag_text == tag_text)).first()
        if tag is None:
            result = connection.execute(TAG.insert(),
                                        tag_text=tag_text)
            tag_id = result.inserted_primary_key[0]
        else:
            tag_id = tag[TAG.c.tag_id]

        tag_ids.add(tag_id)

# Add the tag ids
for tag_id in tag_ids:
    connection.execute(TAG_REGISTER.insert(),
                       tag_id=tag_id,
                       register_id=register_id)
    
for galaxy in galaxies_to_register:
    for unreg_galaxy in galaxy_direct:
        if galaxy in unreg_galaxy:
            result = connection.execute(REGISTER.insert(),
                                    galaxy_name=galaxy.galaxy_name,
                                    galaxy_name_sanitized=sanitize_galaxy_name(galaxy.galaxy_name),
                                    redshift=correct_redshift(galaxy.redshift),
                                    galaxy_type=galaxy.redshift,
                                    filename=sanatize_galaxy_name(galaxy.galaxy_name),
                                    priority=PRIORITY,
                                    register_time=datetime.now(),
                                    run_id=RUN_ID,
                                    sigma=sigma,
                                    sigma_filename=sigma_filename)
            
            shutil.copy(unreg_galaxy, final_path)
            os.remove(unreg_galaxy)
transaction.commit()

LOG.info('Registered %s %s %f %s %d %d', GALAXY_NAME, GALAXY_TYPE, FIXED_GALAXY_NAME, REDSHIFT, INPUT_FILE, PRIORITY, RUN_ID)
for tag_text in TAGS:
    LOG.info('Tag: {0}'.format(tag_text))
