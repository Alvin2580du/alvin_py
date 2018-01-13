# coding:utf-8
"""
Immsg model
Author: alvin
"""
from pyduyp.datasources.elasticsearch import sqldata, sqlresponse
from pyduyp.logger.log import log
import datetime
import pandas as pd
from pyduyp.utils.utils import getespages
from pyduyp.datasources.redis_db import redis
from pyduyp.datasources.immsg import get_rows_by_step


index = 'immsg_all'


def get_message_by_from_to(id_from, id_to, today):
    fielday = datetime.timedelta(days=5)
    fromday = today - fielday
    today = today + fielday
    todayformat = today.strftime('%Y-%m-%d')
    fromdayformat = fromday.strftime('%Y-%m-%d')
    sql = 'select * from {} where createTime>= "{}T00:00:00+00:00" and createTime < "{}T00:00:00+00:00" and  (tenant="{}" and landlord="{}") or (landlord="{}" and tenant="{}") and msgType="text" order by id asc'.format(
        index, fromdayformat, todayformat, id_from, id_to, id_to, id_from)
    log.debug("get from to query es sql:\n {}".format(sql))
    return sqldata(sql)


def get_all_user_by_day(date):
    if date == None:
        today = datetime.date.today()
    else:
        today = date
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    todayformat = today.strftime('%Y-%m-%d')
    yesterdayformat = yesterday.strftime('%Y-%m-%d')
    sqlcount = 'select count(*) as count from {}  where createTime>= "{}T00:00:00+00:00" and createTime < "{}T00:00:00+00:00"'.format(
        index, yesterdayformat, todayformat)
    log.debug(sqlcount)
    countres = sqlresponse(sqlcount)
    log.debug(countres)

    sql = 'select `from`,`to` from {}  where createTime>= "{}T00:00:00+00:00" and createTime < "{}T00:00:00+00:00" order by id asc limit '.format(
        index, yesterdayformat, todayformat)
    pagesize = 1000
    total = int(countres['total'])
    limitindex = getespages(total, pagesize)
    data = []
    i = 0
    for limit in limitindex:
        ids = sqldata(sql + limit)
        # log.debug(ids)
        data.extend(ids)
        i += 1
        if i > 3:
            break
    frame = pd.DataFrame(data)
    return frame.drop_duplicates(['from', 'to'])

#
# def get_rows_by_step(lastid,step=30):
#     sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime`, `uniqid`, `channel` from {} where id>{} and id<={} order by id asc".format(tablenamefull, lastid, lastid + step)
#     log.info(sql)
#     rows = db_session.execute(sql).fetchall()
#     return rows
#
#
# def get_rows_sync_maxid():
#     sql = "select `id` from {}  order by id asc limit 1".format(tablename)
#     log.info(sql)
#     row = db_session.execute(sql).first()
#     return row[0]
#
#
# def get_full_maxid():
#     sql = "select `id` from {}  order by id DESC limit 1".format(tablenamefull)
#     log.info(sql)
#     row = db_session.execute(sql).first()
#     return row[0]


def sync_fullimmsg():
    sync2es_lastkey = 'sync2es_lastkey_full'
    val = redis.get(sync2es_lastkey)
    log.info(val)
    if val != None:
        startid = int(val)
    else:
        startid = 10000
    step = 10000
    datas = get_rows_by_step(startid, step)
    from pyduyp.datasources.elasticsearch import es
    es = es()
    type = 'full'
    if datas and len(datas) > 0:
        for row in datas:
            rowinfo = dict(row.items())
            rowinfo['tenant'] = str(rowinfo['from'])
            rowinfo['landlord'] = str(rowinfo['to'])
            es.index(index=index, doc_type=type, id=rowinfo['id'], body=rowinfo)
        # save last
        startid = dict(datas[-1].items())['id']
    else:
        startid += step
    log.info("lastid:{}".format(startid))
    redis.set(sync2es_lastkey, startid)
    return startid



