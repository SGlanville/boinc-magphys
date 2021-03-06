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
Job to assign any credits in the "credited_job" table to the "area_user" table if not present,
and to remove the rows from "credited_job" table once they have been assigned to the area.
"""
import os
import sys

# Setup the Python Path as we may be running this via ssh
base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..')))
sys.path.append(os.path.abspath(os.path.join(base_path, '../../../../boinc/py')))

import boinc_path_config
from utils.logging_helper import config_logger
from Boinc import database, configxml, db_base
from sqlalchemy.engine import create_engine
from sqlalchemy.sql import select
from config import DB_LOGIN
from sqlalchemy import and_
from database.database_support_core import AREA_USER, AREA, GALAXY_USER

LOG = config_logger(__name__)
LOG.info('PYTHONPATH = {0}'.format(sys.path))

class AssignCredit:
    def __init__(self):
        # Login is set in the database package
        engine = create_engine(DB_LOGIN)
        self._connection = engine.connect()

    def parse_args(self, args):
        """
        Parses arguments provided on the command line and sets
        those argument values as member variables. Arguments
        are parsed as their true types, so integers will be ints,
        not strings.
        """

        args.reverse()
        while len(args):
            arg = args.pop()
            if arg == '-app':
                arg = args.pop()
                self.appname = arg
            else:
                LOG.critical("Unrecognized arg: %s\n", arg)

    def run(self):
        self.parse_args(sys.argv[1:])
        self.config = configxml.default_config().config

        # retrieve app where name = app.name
        database.connect()
        database.Apps.find1(name=self.appname)

        conn = db_base.get_dbconnection()

        user_count = 0
        credit_count = 0

        transaction = self._connection.begin()

        query_cursor = conn.cursor()
        delete_cursor = conn.cursor()
        query_cursor.execute('select userid, workunitid from credited_job ')
        results = query_cursor.fetchall()
        for result in results:
            user_count += 1
            user_id = result['userid']
            area_id = result['workunitid']
            area_user = self._connection.execute(select([AREA_USER]).where(and_(AREA_USER.c.userid == user_id, AREA_USER.c.area_id == area_id))).first()
            if area_user is None:
                area = self._connection.execute(select([AREA]).where(AREA.c.area_id == area_id)).first()
                galaxy_id = area['galaxy_id']
                if area is None:
                    print 'Area', area_id, 'not found, User', user_id, 'not Credited'
                else:
                    AREA_USER.insert().values(userid=user_id, area_id=area_id)
                    GALAXY_USER.insert().prefix_with('IGNORE').values(userid=user_id, galaxy_id=galaxy_id)
                    print 'User', user_id, 'Credited for Area', area_id
                    credit_count += 1

            delete_cursor.execute("delete from credited_job where userid = " + str(user_id) + "  and workunitid = " + str(area_id))

        transaction.commit()
        conn.commit()
        database.close()
        self._connection.close()
        print user_count, 'Users', credit_count, 'Credited'

if __name__ == '__main__':
    assign = AssignCredit()
    assign.run()
