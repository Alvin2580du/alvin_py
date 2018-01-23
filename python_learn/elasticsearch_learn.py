from elasticsearch import Elasticsearch
import requests
import json
from urllib.parse import quote


from pyduyp.decorators.decorator import es_get_decorator
from pyduyp.logger.log import log
from pyduyp.config.conf import get_es_args
print(log)

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
    return Elasticsearch(hosts=[{"host": es_args.get('host'), "port": es_args.get('port')}])


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


@es_get_decorator
def search_source(index, query):
    ret = es().search(index=index, body=query)
    # log.debug(ret)
    res = []
    if ret and len(ret) > 0 and ret['hits']:
        count = ret['hits']['total']
        for hit in ret['hits']['hits']:
            res.append(hit['_source'])
    return res, count


#   setting = {"number_of_shards": 6, "number_of_replicas": 0}
#    mapping = {"timestamp": {"enabled": "true"},
#               "properties": {"logdate": {"type": "date", "format": "yyyy/MM/dd HH:mm:ss", "analyzer": "ik_smart",
#     "search_analyzer": "ik_smart"}}}


# init the index
@es_get_decorator
def create_index(index, settings, mapping, doc_type='doc'):
    settings = {"settings": settings, "mapping": mapping}
    url = "{}://{}:{}/{}".format(es_args.get('schema'), es_args.get('host'), es_args.get('port'), index)
    ret = requests.put(url)
    log.debug("create index request sql url: {} result: {}".format(url, str(ret)))
    mapstr = json.dumps(mapping)
    url += '/{}/_mapping'.format(doc_type)
    log.debug("put {} body: {}".format(url, mapstr))
    ret = requests.put(url, data=mapstr)
    # ret = es().indices.create(index=index, ignore=400, body=settings)
    return ret


def getbyid(index, doc_type='fulltext', id=id):
    try:
        ret = es().get_source(index, doc_type=doc_type, id=id)
        # log.debug(ret)
    except Exception as e:
        return None
    return ret

