# coding:utf-8
"""
config for project
Author: alvin
"""
import os
import random
from yaml import load

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')


with open(config_path, encoding='utf-8') as f:
    cont = f.read()


cf = load(cont)


def set_conf(path):
    global cf
    with open(path, encoding='utf-8') as f:
        content = f.read()
        cf = load(content)
        from pyduyp.logger.log import log
        log.debug("init customize config {} load ok!".format(path))


def get_dictionary():
    return cf.get('dictionary')


def get_db_args():
    return cf.get('db')


def get_dbai_args():
    return cf.get('dbai')


def get_redis_args():
    return cf.get('redis')


def get_neo4j_args():
    return cf.get('neo4j')


def get_es_args():
    return cf.get('elasticsearch')


def get_mq_args():
    return cf.get('rabbitmq')


def get_timeselect_args():
    return cf.get('timeselect')


def get_log_args():
    return cf.get('log')


def get_celery_args():
    return cf.get('celery')


def get_serving():
    return cf.get('serving')


def get_timeout():
    return cf.get('time_out')


def get_product_mode():
    return cf.get('product_mode')


def get_interal():
    interal = random.randint(cf.get('min_interal'), cf.get('max_interal'))
    return interal


def get_exp_interal():
    return cf.get('excp_interal')


def get_redis_url(types):
    redis_info = cf.get('redis')
    host = redis_info.get('host')
    port = redis_info.get('port')
    password = redis_info.get('password')

    if types == 1:
        db = redis_info.get('broker')
    else:
        db = redis_info.get('backend')
    url = 'redis://:{}@{}:{}/{}'.format(password, host, port, db)
    return url


def get_email_args():
    return cf.get('email')


def get_broker_or_backend(types):
    """
    :param types: 类型，1表示中间人，2表示消息后端
    :return: 
    """
    redis_info = cf.get('redis')
    host = redis_info.get('host')
    port = redis_info.get('port')
    password = redis_info.get('password')

    if types == 1:
        db = redis_info.get('broker')
    else:
        db = redis_info.get('backend')
    url = 'redis://:{}@{}:{}/{}'.format(password, host, port, db)

    return url
