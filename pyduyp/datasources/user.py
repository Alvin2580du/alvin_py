# coding:utf-8
"""
User model
Author: alvin
"""
from pyduyp.logger.log import log
from pyduyp.datasources.basic_db import db_session
from pyduyp.decorators.decorator import db_commit_decorator

tablename = 'user'


@db_commit_decorator
def get_user_by_uid(uid):
    sql = "select * from {} where id={}".format(
        tablename, uid)
    log.info("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    # log.debug(row)
    if row != None:
        return dict(row[0].items())
    db_session.close()
    return None

