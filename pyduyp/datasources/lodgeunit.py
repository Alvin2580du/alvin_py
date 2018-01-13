from pyduyp.datasources.basic_db import db_session
from pyduyp.decorators.decorator import db_commit_decorator
from pyduyp.logger.log import log

tablename = 'lodgeunit'


@db_commit_decorator
def get_lodgeunit_by_id(id):
    sql = 'select * from {} where id = {}'.format(tablename, id)
    log.debug("run sql: {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_facilities_from_lodgeunit():
    sql = 'select id, ownerid, facilities from {}'.format(tablename)
    log.debug("run sql: {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_facilities_from_lodgeunit_test():
    sql = 'select id, ownerid, facilities from {} limit 10'.format(tablename)
    log.debug("run sql: {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_facilities_from_lodgeunit_by_id(id, limit=False):
    if limit:
        sql = 'select id, ownerid, facilities from {} where id = {} limit 10'.format(tablename, id)
        log.debug("run sql: {}".format(sql))
    else:
        sql = 'select id, ownerid, facilities from {} where id = {}'.format(tablename, id)
    return db_session.execute(sql).fetchall()
