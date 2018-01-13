# coding=utf-8

"""
create mysql use sqlalchemy
Author: alvin
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pyduyp.config.conf import get_db_args, get_dbai_args


def get_engine(dbargs):
    args = get_db_args()
    if dbargs == 'dbai':
        args = get_dbai_args()
    connect_str = "{}+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(args['db_type'], args['user'], args['password'],
                                                                    args['host'], args['port'], args['db_name'])
    engine = create_engine(connect_str, encoding='utf-8')
    return engine


def get_db(dbargs='default'):
    eng = get_engine(dbargs)
    Session = sessionmaker(bind=eng)
    db_session = Session()
    return db_session


Base = declarative_base()
eng = get_engine('default')
db_session = get_db('default')
dbai_session = get_db('dbai')
metadata = MetaData(get_engine('default'))

__all__ = ['eng', 'Base', 'db_session', 'metadata', 'dbai_session']
