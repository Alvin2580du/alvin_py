# coding:utf-8
"""
create redis client
Author: alvin
"""

import redis
from pyduyp.logger.log import log
from pyduyp.config.conf import get_redis_args

redis_args = get_redis_args()


# redis client
class redis(object):
    rd_con = redis.StrictRedis(host=redis_args.get('host'), port=redis_args.get('port'),
                               password=redis_args.get('password'), db=redis_args.get('db'))

    @classmethod
    def set(cls, key, value):
        log.debug("set to redis,key: {} value: {}".format(key, value))
        cls.rd_con.set(key, value)

    @classmethod
    def get(cls, key):
        return cls.rd_con.get(key)

    @classmethod
    def lpush(cls, key, val):
        return cls.rd_con.lpush(key, val)

    @classmethod
    def rpop(cls, key):
        return cls.rd_con.rpop(key)

    @classmethod
    def instance(cls):
        return cls.rd_con
