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
The routines used to archive the data
"""
from sqlalchemy import create_engine
from archive.archive_boinc_stats_mod import archive_boinc_stats
from archive.archive_hdf5_mod import archive_to_hdf5
from archive.delete_galaxy_mod import delete_galaxy_data
from archive.processed_galaxy_mod import processed_data
from archive.store_files_mod import store_files
from utils.logging_helper import config_logger
from config import DB_LOGIN, ARCHIVE_DATA, ARCHIVE_DATA_DICT
from utils.ec2_helper import EC2Helper

LOG = config_logger(__name__)

USER_DATA = '''#!/bin/bash

# Sleep for a while to let everything settle down
sleep 10s

# Has the NFS mounted properly?
if [ -d '/home/ec2-user/boinc-magphys/server' ]
then
    # We are root so we have to run this via sudo to get access to the ec2-user details
    su -l ec2-user -c 'python2.7 /home/ec2-user/boinc-magphys/server/src/archive/archive_task.py ami'
fi

# All done terminate
shutdown -h now
'''


def process_boinc():
    """
    We're running the process on the BOINC server.

    Check if an instance is still running, if not start it up.
    :return:
    """
    # This relies on a ~/.boto file holding the '<aws access key>', '<aws secret key>'
    ec2_helper = EC2Helper()

    if ec2_helper.boinc_instance_running(ARCHIVE_DATA):
        LOG.info('A previous instance is still running')
    else:
        LOG.info('Starting up the instance')
        instance_type = ARCHIVE_DATA_DICT['instance_type']
        max_price = float(ARCHIVE_DATA_DICT['price'])
        if instance_type is None or max_price is None:
            LOG.error('Instance type and price not set up correctly')
        else:
            bid_price, subnet_id = ec2_helper.get_cheapest_spot_price(instance_type, max_price)
            if bid_price is not None and subnet_id is not None:
                ec2_helper.run_spot_instance(bid_price, subnet_id, USER_DATA, ARCHIVE_DATA, instance_type)
            else:
                ec2_helper.run_instance(USER_DATA, ARCHIVE_DATA, instance_type)


def process_ami():
    """
    We're running on the AMI instance - so actually do the work

    Find the files and move them to S3
    :return:
    """
    # Connect to the database - the login string is set in the database package
    ENGINE = create_engine(DB_LOGIN)
    connection = ENGINE.connect()
    try:
        # Check the processed data
        try:
            processed_data(connection)
        except:
            LOG.exception('processed_data(): an exception occurred')

        # Store files
        try:
            store_files(connection)
        except:
            LOG.exception('store_files(): an exception occurred')

        # Delete galaxy data - commits happen inside
        try:
            delete_galaxy_data(connection)
        except:
            LOG.exception('delete_galaxy_data(): an exception occurred')

        # Archive the BOINC stats
        try:
            archive_boinc_stats()
        except:
            LOG.exception('archive_boinc_stats(): an exception occurred')

        # Archive to HDF5
        try:
            archive_to_hdf5(connection)
        except:
            LOG.exception('archive_to_hdf5(): an exception occurred')

    finally:
        connection.close()
