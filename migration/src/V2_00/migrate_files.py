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
Migrate files to new locations
"""
import glob
import logging
import os
import shlex
import tempfile
from sqlalchemy import select, and_
import urllib
import subprocess
from database.database_support_core import GALAXY
from utils.name_builder import get_galaxy_image_bucket, get_files_bucket, get_galaxy_file_name, get_key_fits, get_thumbnail_colour_image_key, get_colour_image_key, get_build_png_name, get_key_hdf5
from utils.s3_helper import S3Helper
from V2_00 import DRY_RUN

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:' + logging.BASIC_FORMAT)


def get_name_version(full_file_name):
    """
    >>> get_name_version('/home/ec2-user/galaxyImages/239/PGC042868_1.fits')
    ('PGC042868', 1, '.fits')
    >>> get_name_version('/home/ec2-user/galaxyImages/ea/NGC7181_2.fits')
    ('NGC7181', 2, '.fits')
    >>> get_name_version('/home/ec2-user/galaxyImages/239/PGC1191673e_1_tn_colour_1.png')
    ('PGC1191673e', 1, '.png')
    >>> get_name_version('/home/ec2-user/galaxyImages/239/NGC3583_colour_2.png')
    ('NGC3583', 1, '.png')
    >>> get_name_version('/home/ec2-user/galaxyImages/1d1/NGC4269_2_ldust.png')
    ('NGC4269', 2, '.png')
    """
    (head, tail) = os.path.split(full_file_name)
    (filename, extension) = os.path.splitext(tail)
    if extension == '.fits':
        index = filename.rfind('_')
        if index == -1:
            return filename, 1, extension
        else:
            return filename[:index], int(filename[index + 1:]), extension
    else:
        index = filename.find('_1_')
        if index > 1:
            return filename[:index], 1, extension

        index = filename.find('_2_')
        if index > 1:
            return filename[:index], 2, extension

        index = filename.find('_3_')
        if index > 1:
            return filename[:index], 3, extension

        index = filename.find('_4_')
        if index > 1:
            return filename[:index], 4, extension

        index = filename.find('_')
        if index > 1:
            return filename[:index], 1, extension

    return None, None, None


def migrate_image_files(connection, image_bucket_name, file_bucket_name, s3helper):
    for file_name in glob.glob('/home/ec2-user/galaxyImages/*/*'):
        (name, version, extension) = get_name_version(file_name)

        # Only migrate the images in the original galaxy is still in the database
        galaxy = connection.execute(select([GALAXY]).where(and_(GALAXY.c.name == name, GALAXY.c.version_number == version))).first()
        if galaxy is not None:
            if extension == '.fits':
                add_file_to_bucket1(file_bucket_name, get_key_fits(galaxy[GALAXY.c.name], galaxy[GALAXY.c.run_id], galaxy[GALAXY.c.galaxy_id]), file_name, s3helper)
            else:
                galaxy_key = get_galaxy_file_name(galaxy[GALAXY.c.name], galaxy[GALAXY.c.run_id], galaxy[GALAXY.c.galaxy_id])
                if file_name.endswith('_tn_colour_1.png'):
                    add_file_to_bucket1(image_bucket_name, get_thumbnail_colour_image_key(galaxy_key, 1), file_name, s3helper)
                elif file_name.endswith('_colour_1.png'):
                    add_file_to_bucket1(image_bucket_name, get_colour_image_key(galaxy_key, 1), file_name, s3helper)
                elif file_name.endswith('_colour_2.png'):
                    add_file_to_bucket1(image_bucket_name, get_colour_image_key(galaxy_key, 2), file_name, s3helper)
                elif file_name.endswith('_colour_3.png'):
                    add_file_to_bucket1(image_bucket_name, get_colour_image_key(galaxy_key, 3), file_name, s3helper)
                elif file_name.endswith('_colour_4.png'):
                    add_file_to_bucket1(image_bucket_name, get_colour_image_key(galaxy_key, 4), file_name, s3helper)
                elif file_name.endswith('_mu.png'):
                    add_file_to_bucket1(image_bucket_name, get_build_png_name(galaxy_key, 'mu'), file_name, s3helper)
                elif file_name.endswith('_m.png'):
                    add_file_to_bucket1(image_bucket_name, get_build_png_name(galaxy_key, 'm'), file_name, s3helper)
                elif file_name.endswith('_ldust.png'):
                    add_file_to_bucket1(image_bucket_name, get_build_png_name(galaxy_key, 'ldust'), file_name, s3helper)
                elif file_name.endswith('_sfr.png'):
                    add_file_to_bucket1(image_bucket_name, get_build_png_name(galaxy_key, 'sfr'), file_name, s3helper)


def get_temp_file(extension):
    """
    Get a temporary file
    """
    tmp = tempfile.mkstemp(extension, 'pogs', None, False)
    tmp_file = tmp[0]
    os.close(tmp_file)
    return tmp[1]


def check_results(output, path_name):
    """
    Check the output from the command to get a file

    :param output: the text
    :param path_name: the file that should have been written
    :return: True if the files was downloaded correctly
    """
    print('checking file transfer')
    if output.find('HTTP request sent, awaiting response... 200 OK') >= 0 \
            and output.find('Length:') >= 0 \
            and output.find('Saving to:') >= 0 \
            and os.path.exists(path_name) \
            and os.path.getsize(path_name) > 100:
        return True
    return output


def add_file_to_bucket1(file_bucket_name, key, path_name, s3helper):
    if DRY_RUN:
        LOG.info('DRY_RUN: bucket: {0}, key: {1}, file: {2}'.format(file_bucket_name, key, path_name))
    else:
        LOG.info('bucket: {0}, key: {1}, file: {2}'.format(file_bucket_name, key, path_name))
        s3helper.add_file_to_bucket(file_bucket_name, key, path_name)


def migrate_hdf5_files(connection, file_bucket_name, s3helper):
    for galaxy in connection.execute(select([GALAXY])):
        # Get the hdf5 file
        if galaxy[GALAXY.c.version_number] > 1:
            ngas_file_name = '{0}_V{1}.hdf5'.format(galaxy[GALAXY.c.name], galaxy[GALAXY.c.version_number])
        else:
            ngas_file_name = '{0}.hdf5'.format(galaxy[GALAXY.c.name])
        path_name = get_temp_file('hdf5')
        command_string = 'wget -O {0} http://cortex.ivec.org:7780/RETRIEVE?file_id={1}&processing=ngamsMWACortexStageDppi'.format(path_name, urllib.quote(ngas_file_name, ''))
        print(command_string)
        try:
            output = subprocess.check_output(shlex.split(command_string), stderr=subprocess.STDOUT)
            if check_results(output, path_name):
                add_file_to_bucket1(file_bucket_name, get_key_hdf5(galaxy[GALAXY.c.name], galaxy[GALAXY.c.run_id], galaxy[GALAXY.c.galaxy_id]), path_name, s3helper)
            else:
                LOG.error('Big error with {0}'.format(ngas_file_name))
        except subprocess.CalledProcessError as e:
            LOG.exception('Big error')
            raise


def migrate_files(connection):
    """
    Migrate the various files to S3
    """
    LOG.info('Migrating the files')

    s3helper = S3Helper()

    migrate_image_files(connection, get_galaxy_image_bucket(), get_files_bucket(), s3helper)
    migrate_hdf5_files(connection, get_files_bucket(), s3helper)
