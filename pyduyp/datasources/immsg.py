# coding:utf-8
"""
Immsg model
Author: alvin
"""
from pyduyp.datasources.models import Immsg, Immsg_30
from pyduyp.datasources.basic_db import db_session, dbai_session
from pyduyp.decorators.decorator import db_commit_decorator
from pyduyp.logger.log import log
from pyduyp.datasources.redis_db import redis
import datetime

tablenamefull = 'im_msgs'
tablename = 'im_msgs_30'
tablenamelite = 'im_msgs_2'


@db_commit_decorator
def get_message_by_id(id):
    return db_session.execute('select * from {} where id = {}'.format(tablenamelite, id)).fetchall()


@db_commit_decorator
def get_message_by_from_to(id_from, id_to):
    return db_session.execute('select * from user where id = 1')


@db_commit_decorator
def get_message_by_from_to(id_from, id_to, today, days=3):
    fielday = datetime.timedelta(days=days)
    fromday = today - fielday
    today = today + fielday
    todayformat = today.strftime('%Y-%m-%d')
    fromdayformat = fromday.strftime('%Y-%m-%d')
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where `from`={} and `to`={} || `from`= {} and `to`= {} and (createTime>'{} 00:00:00' and createTime<='{} 00:00:00') order by id asc".format(
        tablenamelite, id_from, id_to, id_to, id_from, fromdayformat, todayformat)
    # log.debug("get from to query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_message_by_from_to_full(id_from, id_to, today):
    fielday = datetime.timedelta(days=5)
    fromday = today - fielday
    today = today + fielday
    todayformat = today.strftime('%Y-%m-%d')
    fromdayformat = fromday.strftime('%Y-%m-%d')
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where (createTime>'{} 00:00:00' and createTime<='{} 00:00:00') and `from`={} and `to`={} || `from`= {} and `to`= {}  order by id asc".format(
        tablenamefull, fromdayformat, todayformat, id_from, id_to, id_to, id_from)
    log.debug("get from to query sql:\n {}".format(sql))
    return dbai_session.execute(sql).fetchall()


@db_commit_decorator
def get_all_user_by_day(date):
    if date == None:
        today = datetime.date.today()
    else:
        today = date
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    todayformat = today.strftime('%Y-%m-%d')
    yesterdayformat = yesterday.strftime('%Y-%m-%d')
    sql = "select CAST(`from` AS CHAR)+CAST(`to` AS CHAR) as ids ,`from`,`to` from {}  where createTime>'{} 00:00:00' and createTime<='{} 00:00:00' group by ids order by id asc;".format(
        tablename, yesterdayformat, todayformat)
    log.info("get query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_all_user_by_day_full(date):
    if date == None:
        today = datetime.date.today()
    else:
        today = date
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    todayformat = today.strftime('%Y-%m-%d')
    yesterdayformat = yesterday.strftime('%Y-%m-%d')
    sql = "select CAST(`from` AS CHAR)+CAST(`to` AS CHAR) as ids ,`from`,`to` from {}  where createTime>'{} 00:00:00' and createTime<='{} 00:00:00' group by ids order by id asc;".format(
        tablenamefull, yesterdayformat, todayformat)
    log.debug("get query sql:\n {}".format(sql))
    return dbai_session.execute(sql).fetchall()


@db_commit_decorator
def get_all_user_by_day_limit(date, limit=5):
    if date == None:
        today = datetime.date.today()
    else:
        today = date
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    todayformat = today.strftime('%Y-%m-%d')
    yesterdayformat = yesterday.strftime('%Y-%m-%d')
    sql = "select CAST(`from` AS CHAR)+CAST(`to` AS CHAR) as ids ,`from`,`to` from {}  where createTime>'{} 00:00:00' and createTime<='{} 00:00:00' group by ids order by id asc limit {};".format(
        tablename, yesterdayformat, todayformat, limit)
    log.info("get query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_all_user_by_day_limit_full(date, limit=5):
    if date == None:
        today = datetime.date.today()
    else:
        today = date
    oneday = datetime.timedelta(days=1)
    yesterday = today - oneday
    todayformat = today.strftime('%Y-%m-%d')
    yesterdayformat = yesterday.strftime('%Y-%m-%d')
    sql = "select CAST(`from` AS CHAR)+CAST(`to` AS CHAR) as ids ,`from`,`to` from {}  where createTime>'{} 00:00:00' and createTime<='{} 00:00:00' group by ids order by id asc limit {};".format(
        tablenamefull, yesterdayformat, todayformat, limit)
    log.debug("get query sql:\n {}".format(sql))
    return dbai_session.execute(sql).fetchall()


@db_commit_decorator
def get_rows_by_step(lastid, step=30):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime`, `uniqid`, `channel` from {} where id>{} and id<={} order by id asc".format(
        tablename, lastid, lastid + step)
    log.debug(sql)
    rows = db_session.execute(sql).fetchall()
    return rows


@db_commit_decorator
def get_ids_by_step_full(lastid, step=30):
    sql = "select `id`, `from`, `to`, `userType`, `createTime` from {} where id>{} and id<={} order by id asc".format(
        tablenamefull, lastid, lastid + step)
    log.debug(sql)
    rows = dbai_session.execute(sql).fetchall()
    return rows


@db_commit_decorator
def get_rows_sync_maxid():
    sql = "select `id` from {}  order by id desc limit 1".format(tablename)
    log.debug(sql)
    row = db_session.execute(sql).first()
    return row[0]


@db_commit_decorator
def get_full_maxid():
    sql = "select `id` from {}  order by id DESC limit 1".format(tablenamefull)
    log.debug(sql)
    row = dbai_session.execute(sql).first()
    return row[0]


# get all rows backup
@db_commit_decorator
def get_full_rows_by_step(lastid, step=30):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime`, `uniqid`, `channel` from {} where id>{} and id<={} order by id asc".format(
        tablenamefull, lastid, lastid + step)
    rows = dbai_session.execute(sql).fetchall()
    log.info("run ai db: {}".format(sql))
    return rows


@db_commit_decorator
def sync_30tofull():
    startid = get_full_maxid()
    log.debug("get ai full maxid: {}".format(startid))
    step = 100000
    datas = get_rows_by_step(startid, step)

    if datas and len(datas) > 0:
        log.debug("get datas count: {}".format(len(datas)))
        instances = []
        for row in datas:
            rowinfo = dict(row.items())
            # write to db
            immsg = Immsg(rowinfo)
            instances.append(immsg)
        log.debug("get instance len: {}".format(len(instances)))
        dbai_session.add_all(instances)
        dbai_session.commit()
        # save last
        startid = dict(datas[-1].items())['id']
    else:
        startid += step
    log.debug("lastid:{}".format(startid))
    return startid


# get immsg user pairs
def sync_session_userpairs():
    # 总共有多少对
    session_userpairs_total = 's:msg:userpairs:total'
    # 每天有多少对
    session_userpairs_day = 's:msg:userpairs:date:{}'
    # 每对的消息的消息ids
    session_userpairs_from_to = 's:msg:userpairs:ids:{}:{}'
    # 房客每天聊几个房东
    session_from_total = 's:msg:from:total:{}'
    session_to_total = 's:msg:to:total:{}'
    session_from_date = 's:msg:from:date:{}:{}'
    session_to_date = 's:msg:to:date:{}:{}'

    sync_lastidkey = 's:msg:sync:lastid'
    val = redis.get(sync_lastidkey)
    log.info(val)
    if val != None:
        startid = int(val)
    else:
        startid = 10000
    step = 20000
    datas = get_ids_by_step_full(startid, step)
    log.debug("get total datas len: {}".format(len(datas)))

    if datas and len(datas) > 0:
        for row in datas:
            rowinfo = dict(row.items())
            if rowinfo['userType'] == 'landlord':
                tmp = rowinfo['from']
                rowinfo['from'] = rowinfo['to']
                rowinfo['to'] = tmp
            date = rowinfo['createTime'].strftime('%Y-%m-%d')
            # log.debug("get createtime: {}".format(date))
            redis.instance().sadd(session_userpairs_total, '{}:{}'.format(rowinfo['from'], rowinfo['to']))
            redis.instance().sadd(session_userpairs_day.format(date), '{}:{}'.format(rowinfo['from'], rowinfo['to']))
            redis.instance().sadd(session_userpairs_from_to.format(rowinfo['from'], rowinfo['to']), rowinfo['id'])

            redis.instance().sadd(session_from_total.format(rowinfo['from']), rowinfo['to'])
            redis.instance().sadd(session_to_total.format(rowinfo['to']), rowinfo['from'])
            redis.instance().sadd(session_from_date.format(rowinfo['from'], date), rowinfo['to'])
            redis.instance().sadd(session_to_date.format(rowinfo['to'], date), rowinfo['from'])
        # save last
        startid = dict(datas[-1].items())['id']
    else:
        startid += step
    log.debug("lastid:{}".format(startid))
    redis.set(sync_lastidkey, startid)
    return startid


@db_commit_decorator
def get_message_by_ids(ids):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where id in ({}) order by id asc".format(
        tablenamefull, ids)
    # log.debug("get from to query sql:\n {}".format(sql))
    return dbai_session.execute(sql).fetchall()


@db_commit_decorator
def get_message_by_ids_lite(ids):
    sql = "select `id`, `from`, `to`, `userType`, `msgType`, `msg` from {} where id in ({}) and msgType='text' order by id asc".format(
        tablenamefull, ids)
    return dbai_session.execute(sql).fetchall()


@db_commit_decorator
def get_all_user_by_day_msg_2():
    sql = "select CAST(`from` AS CHAR)+CAST(`to` AS CHAR) as ids ,`from`,`to`,createTime,msg,msgType,roomId from {}  group by ids order by id asc".format(
        tablenamelite)
    log.info("get query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_all_user_by_day_msg_2_limit(limit):
    sql = "select CAST(`from` AS CHAR)+CAST(`to` AS CHAR) as ids ,`from`,`to`,createTime,msg,msgType,roomId from {}  group by ids order by id asc limit {}".format(
        tablenamelite, limit)
    log.info("get query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_last_id_msg_2():
    sql = "select id from {} order by id desc limit 1".format(
        tablenamelite)
    log.info("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    if row:
        return dict(row[0].items())['id']
    return None


@db_commit_decorator
def get_start_id_msg_by_date_30(date):
    sql = "select id from {} where DATE_FORMAT(createTime, '%Y-%m-%d %H:00:00') = '{}' order by id asc limit 1".format(
        tablename, date)
    log.info("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    if row:
        return dict(row[0].items())['id']
    return None


@db_commit_decorator
def get_start_id_msg_by_date_2_and_user():
    sql = "select id from {} where DATE_FORMAT(createTime, '%Y-%m-%d 00:00:00') = '{}' order by id asc limit 1".format(
        tablenamelite, datetime.date.today().strftime('%Y-%m-%d 00:00:00'))
    log.info("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    if row:
        return dict(row[0].items())['id']
    return None


@db_commit_decorator
def get_large_then_id_msg_2(id):
    sql = "select id,`from`,`to`,createTime,msg,msgType,roomId,userType from {} where id>{} order by id asc limit 1".format(
        tablenamelite, id)
    # log.info("get query sql:\n {}".format(sql))
    # if is null then block?
    row = db_session.execute(sql).fetchall()
    # log.debug(row)
    if row != None and len(row) > 0:
        return dict(row[0].items())
    db_session.close()
    return None


@db_commit_decorator
def get_large_then_id_msg_30(id):
    sql = "select id,`from`,`to`,createTime,msg,msgType,roomId,userType from {} where id>{} order by id asc limit 1".format(
        tablename, id)
    # log.info("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    # log.debug(row)
    if row != None:
        return dict(row[0].items())
    db_session.close()
    return None


@db_commit_decorator
def get_large_then_id_in_date_msg_30(id, date):
    sql = "select id,`from`,`to`,createTime,msg,msgType,roomId,userType from {} where id>{} and DATE_FORMAT(createTime, '%Y-%m-%d %H:00:00') = '{}' order by id asc limit 1".format(
        tablename, id, date)
    log.debug("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    # log.debug(row)
    if row != None and len(row) > 0:
        return dict(row[0].items())
    return None


@db_commit_decorator
def get_large_then_id_in_date_msg_2_and_user(id, uid):
    sql = "select id,`from`,`to`,createTime,msg,msgType,roomId,userType from {} where id>{} and `from`={} and DATE_FORMAT(createTime, '%Y-%m-%d 00:00:00') = '{}' order by id asc".format(
        tablename, id, uid, datetime.date.today().strftime('%Y-%m-%d 00:00:00'))
    log.debug("get query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    # log.debug(row)
    if row != None and len(row) > 0:
        return dict(row[0].items())
    return None


@db_commit_decorator
def get_message_by_from_to_before(id_from, id_to, today):
    todayformat = today.strftime('%Y-%m-%d %H:%M:%S')
    # 取3天前的
    oneday = datetime.timedelta(days=3)
    yesterday = today - oneday
    yesterdayformat = yesterday.strftime('%Y-%m-%d 00:00:00')
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where (`from`={} and `to`={} || `from`= {} and `to`= {}) and (createTime>'{}' and createTime<='{}') order by id asc".format(
        tablename, id_from, id_to, id_to, id_from, yesterdayformat, todayformat)
    log.debug("get from to query before sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_message_by_from_to_before_id(id_from, id_to, id):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where `from`={} and `to`={} || `from`= {} and `to`= {} and id<={} order by id asc".format(
        tablename, id_from, id_to, id_to, id_from, id)
    # log.debug("get from to query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


# get by from user
@db_commit_decorator
def get_message_by_from(id_from):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where `from`={}  order by id asc".format(
        tablename, id_from)
    log.debug("get from to query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_message_by_from_in(id_from_arr):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where `from` in ({}) and userType='tenant'  order by id asc".format(
        tablename, ','.join(id_from_arr))
    log.debug("get from to query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


@db_commit_decorator
def get_message_by_from_count(id_from):
    sql = "select count(`id`) as count from {} where `from`={}  order by id asc".format(
        tablename, id_from)
    log.debug("get from to query sql:\n {}".format(sql))
    row = db_session.execute(sql).fetchall()
    if row != None:
        return dict(row[0].items())['count']
    return 0


@db_commit_decorator
def get_message_by_from_time_list(id_from, format="%Y-%m-%d %H:%M:%S"):
    sql = "select `createTime` from {} where `from`={}  order by id asc".format(
        tablename, id_from)
    log.debug("get from to query sql:\n {}".format(sql))
    rows = db_session.execute(sql).fetchall()
    ret = []
    if rows != None:
        for row in rows:
            val = dict(row.items())['createTime'].strftime(format)
            if val not in ret:
                ret.append(val)
    return ret


@db_commit_decorator
def get_message_by_from_time_list_date(id_from, date, format="%Y-%m-%d %H:%M:%S"):
    sql = "select `createTime` from {} where `from`={} and DATE_FORMAT(createTime, '%Y-%m-%d') = '{}' order by id asc".format(
        tablename, id_from, date)
    log.debug("get from to query sql:\n {}".format(sql))
    rows = db_session.execute(sql).fetchall()
    ret = []
    if rows != None:
        for row in rows:
            val = dict(row.items())['createTime'].strftime(format)
            if val not in ret:
                ret.append(val)
    return ret


@db_commit_decorator
def get_message_by_datehour(datehour):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where DATE_FORMAT(createTime, '%Y-%m-%d %H:00:00') = '{}' order by id asc".format(
        tablename, datehour)
    log.debug("get from to query sql:\n {}".format(sql))
    rows = db_session.execute(sql).fetchall()
    return rows


@db_commit_decorator
def get_message_by_id_30(id):
    sql = "select `id`, `from`, `to`, `talkId`, `roomId`, `userType`, `msgType`, `msg`, `platform`, `version`, `state`, `type`, `opType`, `createTime` from {} where `id`={}  order by id asc".format(
        tablename, id)
    log.debug("get from to query sql:\n {}".format(sql))
    return db_session.execute(sql).fetchall()


# get immsg user pairs
def sync_session_user_msg_map():
    user_msgs = 's:common:msg:user:{}'
    sync_lastidkey = 's:common:msg:sync:lastid'
    val = redis.get(sync_lastidkey)
    log.info(val)
    if val != None:
        startid = int(val)
    else:
        startid = 10000
    step = 20000
    datas = get_ids_by_step_full(startid, step)
    log.debug("get total datas len: {}".format(len(datas)))

    if datas and len(datas) > 0:
        for row in datas:
            rowinfo = dict(row.items())
            if rowinfo['userType'] == 'landlord':
                rowinfo['from'] = rowinfo['to']
            # date = rowinfo['createTime'].strftime('%Y-%m-%d')
            # log.debug("get createtime: {}".format(date))
            redis.instance().sadd(user_msgs.format(rowinfo['from']), rowinfo['id'])
        # save last
        startid = dict(datas[-1].items())['id']
    else:
        startid += step
    log.debug("lastid:{}".format(startid))
    redis.set(sync_lastidkey, startid)
    return startid


@db_commit_decorator
def get_before_message_by_iduid(id, uid):
    sql = "select `id`, `msgType`, `msg`, `createTime` from {} where id < {} and `from`={} order by id desc limit 6".format(
        tablename, id, uid)
    log.debug("get from to query sql:\n {}".format(sql))
    rows = db_session.execute(sql).fetchall()
    return rows


@db_commit_decorator
def get_tenant_message_by_ids(ids):
    sql = "select `id`, `from`, `to`, `roomId`, `userType`, `msgType`, `msg`, `createTime` from {}" \
          " where id in ({})  and userType='tenant' order by id asc".format(
        tablenamefull, ids)

    # log.info("get from to query sql:\n {}".format(sql))
    return dbai_session.execute(sql).fetchall()


@db_commit_decorator
def get_full_message_by_id(id):
    return dbai_session.execute('select * from {} where id = {}'.format(tablenamefull, id)).fetchall()


@db_commit_decorator
def get_tenant_message_by_tenant(tenant):
    sql = "select `from`, `userType`, `msgType`, `msg` from {} where `from`={} and userType='tenant' and msgType='text'".format(
        tablename, tenant)
    # log.debug(sql)
    return db_session.execute(sql).fetchall()

@db_commit_decorator
def get_tenant_full_message_by_tenant(tenant):
    sql = "select `from`, `userType`, `msgType`, `msg` from {} where `from`={} and userType='tenant' and msgType='text'".format(
        tablenamefull, tenant)
    # log.debug(sql)
    return db_session.execute(sql).fetchall()