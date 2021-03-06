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
Look at the running galaxies and see if they are done
"""
import os
import sys

# Setup the Python Path as we may be running this via ssh
base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '../../src')))
sys.path.append(os.path.abspath(os.path.join(base_path, '../../../../boinc/py')))

import datetime
from utils.logging_helper import config_logger
from archive.processed_galaxy_mod import sort_data, finish_processing
from sqlalchemy.engine import create_engine
from sqlalchemy.sql.expression import select
from config import BOINC_DB_LOGIN, DB_LOGIN, PROCESSED, COMPUTING
from database.boinc_database_support_core import RESULT
from database.database_support_core import GALAXY

LOG = config_logger(__name__)
LOG.info('PYTHONPATH = {0}'.format(sys.path))

# Get the work units still being processed
ENGINE = create_engine(BOINC_DB_LOGIN)
connection = ENGINE.connect()
current_jobs = []
for result in connection.execute(select([RESULT]).where(RESULT.c.server_state != 5)):
    current_jobs.append(result[RESULT.c.name])
connection.close()

# Connect to the database - the login string is set in the database package
ENGINE = create_engine(DB_LOGIN)
connection = ENGINE.connect()

sorted_data = sort_data(connection, current_jobs)
for key in sorted(sorted_data.iterkeys()):
    LOG.info('{0}: {1} results'.format(key, len(sorted_data[key])))

try:
    # Get the galaxies we know are still processing
    processed = []
    for galaxy in connection.execute(select([GALAXY]).where(GALAXY.c.status_id == COMPUTING)):
        if finish_processing(galaxy[GALAXY.c.name], galaxy[GALAXY.c.galaxy_id], sorted_data):
            processed.append(galaxy[GALAXY.c.galaxy_id])
            LOG.info('%d %s has completed', galaxy[GALAXY.c.galaxy_id], galaxy[GALAXY.c.name])

    transaction = connection.begin()
    for galaxy_id in processed:
        connection.execute(GALAXY.update().where(GALAXY.c.galaxy_id == galaxy_id).values(status_id=PROCESSED, status_time=datetime.datetime.now()))
    transaction.commit()

    LOG.info('Marked %d galaxies ready for archiving', len(processed))
    LOG.info('%d galaxies are still being processed', len(sorted_data))

except Exception:
    LOG.exception('Major error')

finally:
    connection.close()
