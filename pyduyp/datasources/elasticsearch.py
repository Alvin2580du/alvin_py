from elasticsearch import Elasticsearch
from pyduyp.config.conf import get_es_args
import requests
from pyduyp.logger.log import log
import json
from pyduyp.decorators.decorator import es_get_decorator
from urllib.parse import quote

"""
elastic search client for get info

from pandas.io.json import json_normalize
from coinphd.datasources.elasticsearch import sqlresponse

sql = 'select * from bitflyer order by id asc limit 100000000'
    res = sqlresponse(sql)
    df = json_normalize(res['hits'])

"""
es_args = get_es_args()


# get es instance
def es():
    return Elasticsearch([{"host": es_args.get('host'), "port": es_args.get('port')}])


# es run sql
@es_get_decorator
def sql(sql):
    url = "{}://{}:{}/_sql?sql={}".format(es_args.get('schema'), es_args.get('host'), es_args.get('port'), sql)
    log.debug("request sql url: {}".format(url))
    response = requests.get(url)
    return response


# es run sql
@es_get_decorator
def sqlresponse(sql):
    sql = quote(sql)
    log.debug("get encode sql {}".format(sql))
    url = "{}://{}:{}/_sql?sql={}".format(es_args.get('schema'), es_args.get('host'), es_args.get('port'), sql)
    log.debug("request sql url: {}".format(url))
    res = []
    try:
        response = requests.get(url)
        content = response.content.decode('utf8')
    except:
        log.error("request get {} error!".format(url))
        pass

    if len(content) > 2:
        res = json.loads(content)['hits']
    return res


@es_get_decorator
def sqldata(sql):
    log.debug("get encode sql {}".format(sql))
    sql = quote(sql)
    url = "{}://{}:{}/_sql?sql={}".format(es_args.get('schema'), es_args.get('host'), es_args.get('port'), sql)
    log.debug("request sql url: {}".format(url))
    response = requests.get(url)
    content = response.content.decode('utf8')
    res = []
    if len(content) > 2:
        ret = json.loads(content)['hits']
        log.debug(ret)
        if ret and len(ret) > 0 and ret['hits']:
            for hit in ret['hits']:
                res.append(hit['_source'])
    return res


@es_get_decorator
def sqldatacount(sql):
    sql = quote(sql)
    log.debug("get encode sql {}".format(sql))
    url = "{}://{}:{}/_sql?sql={}".format(es_args.get('schema'), es_args.get('host'), es_args.get('port'), sql)
    log.debug("request sql url: {}".format(url))
    response = requests.get(url)
    content = response.content.decode('utf8')
    res = []
    count = 0
    if len(content) > 2:
        ret = json.loads(content)['hits']
        count = ret['count']
        if ret and len(ret) > 0 and ret['hits']:
            for hit in ret['hits']:
                res.append(hit['_source'])
    return res, count


@es_get_decorator
def search(index, query):
    res = es().search(index=index, body=query)
    hits = res['hits']['hits']
    return hits
