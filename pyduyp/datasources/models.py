# coding=utf-8
"""
Base model
Author: alvin
"""
from pyduyp.datasources.basic_db import Base
from pyduyp.datasources.tables import *


class User(Base):
    __table__ = user_info


class Immsg_30(Base):
    __table__ = im_msgs_30_info


class Immsg(Base, dict):
    __table__ = im_msgs_info

    def __init__(self, dic):
        for key, val in dic.items():
            self.__dict__[key] = self[key] = Immsg(val) if isinstance(val, dict) else val


class UserFrozenAccount(Base, dict):
    __table__ = user_frozen_account_info

    def __init__(self, dic):
        for key, val in dic.items():
            self.__dict__[key] = self[key] = UserFrozenAccount(val) if isinstance(val, dict) else val
