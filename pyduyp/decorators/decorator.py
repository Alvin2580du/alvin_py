from functools import wraps

from functools import wraps
from pyduyp.datasources.basic_db import db_session
from pyduyp.logger.log import log


def es_get_decorator(func):
    @wraps(func)
    def es_get(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('es get response error, detail {}'.format(e))

    return es_get


def db_commit_decorator(func):
    @wraps(func)
    def session_commit(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log.error('db operation error, detail {}'.format(e))
            log.warning('transaction rollbacks')
            db_session.rollback()

    return session_commit
