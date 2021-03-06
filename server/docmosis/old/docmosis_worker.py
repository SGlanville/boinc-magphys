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
Docmosis Worker Process - process jobs in docmosis_task table.
"""
import os
import sys
import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s:' + logging.BASIC_FORMAT)

# Setup the Python Path as we may be running this via ssh
base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '../../src')))
sys.path.append(os.path.abspath(os.path.join(base_path, '../../../../boinc/py')))

import uuid
import docmosis_mod
from datetime import datetime
from sqlalchemy.engine import create_engine
from sqlalchemy.sql import select, and_
from config import DB_LOGIN
from database.database_support_core import DOCMOSIS_TASK, DOCMOSIS_TASK_GALAXY

ENGINE = create_engine(DB_LOGIN)


class TaskInfo():
    def __init__(self):
        self.task_id = ""
        self.userid = ""
        self.galaxy_ids = []


def main():
    """
    This is where we'll usually start
    """

    # Generate unique token for this worker instance
    token = uuid.uuid4().hex

    connection = ENGINE.connect()

    assign_tasks(token, connection)
    LOG.info("Worker started with token %s" % token)
    tasks = get_tasks(token, connection)
    LOG.info("Got %d tasks to process" % len(tasks))
    for task in tasks:
        LOG.info("Task id #%s - Processing start" % task.task_id)
        try:
            run_task(task, connection)
            LOG.info("Task id #%s - Completed succesfully" % task.task_id)
        except Exception, e:
            LOG.info("Task id #%s - Failed with error \"%s\"" % (task.task_id, e))
            LOG.exception('Major Error')
            deassign_task(task, connection)
    LOG.info("Worker finished")
    connection.close()


def run_task(task, connection):
    try:
        docmosis_mod.email_galaxy_report(connection, task.userid, task.galaxy_ids)
        query = DOCMOSIS_TASK.update()
        query = query.where(DOCMOSIS_TASK.c.task_id == task.task_id)
        query = query.values(finish_time=datetime.now(), status=2)
        connection.execute(query)
    except:
        raise


def assign_tasks(token, connection):
    query = DOCMOSIS_TASK.update()
    query = query.where(and_(DOCMOSIS_TASK.c.worker_token == None, DOCMOSIS_TASK.c.status == 1))
    query = query.values(worker_token=token)
    connection.execute(query)


def deassign_task(task, connection):
    query = DOCMOSIS_TASK.update()
    query = query.where(DOCMOSIS_TASK.c.task_id == task.task_id)
    query = query.values(worker_token=None, status=1)
    connection.execute(query)


def get_tasks(token, connection):
    query = select([DOCMOSIS_TASK])
    query = query.where(and_(DOCMOSIS_TASK.c.worker_token == token, DOCMOSIS_TASK.c.status == 1))
    tasks = connection.execute(query)
    task_list = []
    for task in tasks:
        task_line = TaskInfo()
        task_line.task_id = task.task_id
        task_line.userid = task.userid
        query = select([DOCMOSIS_TASK_GALAXY])
        query = query.where(DOCMOSIS_TASK_GALAXY.c.task_id == task.task_id)
        galaxies = connection.execute(query)
        for galaxy in galaxies:
            task_line.galaxy_ids.append(galaxy.galaxy_id)
        task_list.append(task_line)

    return task_list


if __name__ == '__main__':
    main()
