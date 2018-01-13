import datetime
from pyduyp.logger.log import log
from pyduyp.datasources.basic_db import db_session, dbai_session, get_db
from pyduyp.decorators.decorator import db_commit_decorator
from pyduyp.datasources.models import UserFrozenAccount
from pyduyp.config.conf import get_product_mode

tablename = 'user_frozen_account'
product_mode = get_product_mode()


def get_user(uid):
    sql = "select * from {} where userid={} limit 1".format(
        tablename, uid)
    log.debug("get query sql:\n {}".format(sql))
    try:
        row = db_session.execute(sql).fetchall()
        # log.debug(row)
        if row is not None and len(row) > 0:
            return dict(row[0].items())
    except Exception as e:
        log.warning("select error: {}".format(str(e)))
    db_session.close()
    return None


# add user to db
@db_commit_decorator
def add_user(uid, user):
    forzentype = ['im']
    user_list = []
    try:
        for t in forzentype:
            forbid_user = {}
            forbid_user['userid'] = uid
            forbid_user['username'] = user['username']
            forbid_user['mobile'] = user['mobile']
            forbid_user['state'] = 1
            forbid_user['operid'] = 2910
            forbid_user['opername'] = 'aibot'
            forbid_user['frozenreason'] = '竞对'
            forbid_user['createtime'] = datetime.datetime.now()
            forbid_user['updatetime'] = datetime.datetime.now()
            forbid_user['frozentype'] = t
            forbid_user['devnum'] = 0
            forbid_user = UserFrozenAccount(forbid_user)
            user_list.append(forbid_user)

        dbai_session.add_all(user_list)
        dbai_session.commit()
    except Exception as e:
        log.warning('add user error: {} {} {}'.format(uid, user['mobile'], str(e)))
        # re connect then to commit
        reconnect_dbai(uid, user)
        return False
    return True


def reconnect_dbai(uid, user):
    global dbai_session
    dbai_session = get_db('dbai')
    log.warning("do reconnect to db, check the connect, retry insert: {}".format(uid))
    add_user(uid, user)

