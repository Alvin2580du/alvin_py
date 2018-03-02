from pyduyp.datasources.basic_db import db_session
from pyduyp.decorators.decorator import db_commit_decorator
from pyduyp.datasources.models import XiaoHua

from pyduyp.logger.log import log
from pyduyp.utils.fileops import curlmd5

table_name = 'xiaohua'


@db_commit_decorator
def getone():
    sql = "select * from {} where status != 1 limit 1".format(table_name)
    try:
        row = db_session.execute(sql).fetchall()
        if row is not None and len(row) > 0:
            return dict(row[0].items())
    except Exception as e:
        log.warning("! xiaohua select error: {}".format(e))
    db_session.close()
    return None


@db_commit_decorator
def add(inputs_dict):
    try:
        add_dict = {"name": inputs_dict['name'],
                    "namemd5": curlmd5(inputs_dict['name']),
                    "status": inputs_dict['status'],
                    "createtime": inputs_dict['createtime']}
        add_obj = XiaoHua(add_dict)
        db_session.add(add_obj)
        db_session.commit()
    except Exception as e:
        log.warning(e)
        add_obj = None
    db_session.close()
    return add_obj


def update(idstr, inputs_dict):
    try:
        update_dict = {"name": inputs_dict['name'],
                       "namemd5": curlmd5(inputs_dict['name']),
                       "status": inputs_dict['status'],
                       "createtime": inputs_dict['createtime']}
        db_session.query(XiaoHua).filter(XiaoHua.id == idstr).update(update_dict)
        db_session.commit()
        db_session.close()
    except Exception as e:
        log.warning(e)
        return None
    return update_dict


@db_commit_decorator
def search_add_change_status(name):
    sql = "select * from {} where namemd5={}".format(table_name, name)
    try:
        row = db_session.execute(sql).fetchall()
        return row
    except Exception as e:
        log.warning("! xiaohua select error")
    db_session.close()
    return None



