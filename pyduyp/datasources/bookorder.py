from pyduyp.datasources.basic_db import db_session
from pyduyp.decorators.decorator import db_commit_decorator
from pyduyp.logger.log import log
from datetime import timedelta
tablename = 'bookorder'


@db_commit_decorator
def get_bookorder_by_id(id):
    sql = 'select * from {} where id = {}'.format(tablename, id)
    log.debug("run sql: {}".format(sql))
    return db_session.execute(sql).fetchall()


def check_bookorder(tanentid, ownerid, lodgeunitid, createtime):
    oneday = timedelta(days=1)
    yesterday = createtime - oneday
    todayformat = createtime.strftime('%Y-%m-%d %H:%M:%S')
    yesterdayformat = yesterday.strftime('%Y-%m-%d %H:%M:%S')
    sql = 'select * from {} where tenantid = {} and ownerid={} and lodgeunitid={} ' \
          'and createtime>="{}" and createtime<="{}"'\
        .format(tablename, tanentid, ownerid, lodgeunitid, yesterdayformat, todayformat)
    log.debug("run sql: {}".format(sql))
    return db_session.execute(sql).fetchall()


