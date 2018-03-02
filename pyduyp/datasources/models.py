from pyduyp.datasources.basic_db import Base

from pyduyp.datasources.tables import *


class XiaoHua(Base, dict):
    __table__ = xiaohua

    def __init__(self, dic):
        for k, v in dic.items():
            self.__dict__[k] = self[k] = XiaoHua(v) if isinstance(v, dict) else v
