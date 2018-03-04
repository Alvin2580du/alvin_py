from pyduyp.datasources.basic_db import db_session
from pyduyp.decorators.decorator import db_commit_decorator
from pyduyp.datasources.models import CaiLianShe

from pyduyp.logger.log import log

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
        add_dict = {"news": inputs_dict['news'],
                    "createtime": inputs_dict['createtime'],
                    "comment": inputs_dict['comment'],
                    }
        add_obj = CaiLianShe(add_dict)
        db_session.add(add_obj)
        db_session.commit()
    except Exception as e:
        log.warning(e)
        add_obj = None
    db_session.close()
    return add_obj


def update(idstr, inputs_dict):
    try:
        update_dict = {"news": inputs_dict['news'],
                       "time": inputs_dict['name'],
                       "status": inputs_dict['status'],
                       "createtime": inputs_dict['createtime'],
                       "comment": inputs_dict['comment'],
                       }
        db_session.query(CaiLianShe).filter(CaiLianShe.id == idstr).update(update_dict)
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
