import requests
import json
from elasticsearch import Elasticsearch
import pandas as pd
from tqdm import tqdm
import os
from elasticsearch.helpers import bulk

from pyduyp.logger.log import log
from pyduyp.utils.utils import replace_symbol
from pyduyp.config.conf import get_es_args
from pyduyp.utils.fileops import curlmd5


es_args = get_es_args()
es = Elasticsearch()


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


def create_entity_tmp():
    setting = {"number_of_shards": 6, "number_of_replicas": 0}
    mapping = {
        "properties":
            {
                "q": {"type": "text"},
                "a": {"type": "text"},
                "roomId": {"type": "text"},
                "tenantId": {"type": "text", "analyzer": "ik_smart", "search_analyzer": "ik_smart"},
                "lanlordId": {"type": "text", "analyzer": "ik_smart", "search_analyzer": "ik_smart"},
                "id": {"type": "text"}
            }
    }
    index_name = 'bot_entity_tmp_new'
    try:
        es.indices.delete(index_name)
    except:
        pass
    ret_entity = create_index(index_name, setting, mapping, doc_type='fulltext')
    log.debug(ret_entity)
    data = pd.read_csv("antbot/datasets/question_45/import_new.csv")
    for message in tqdm(data.values):
        if isinstance(message[0], str) and isinstance(message[1], str):
            body = {'q': message[0], 'a': message[1], 'roomId': message[2],
                    'tenantId': message[3], 'lanlordId': message[4], 'id': str(curlmd5(message[0]))}
            es.index(index_name, doc_type="fulltext", id=body['id'], body=body)


def message_Clustering():
    data = pd.read_csv("antbot/datasets/question_45/root_q_a").values.tolist()
    search_msg = [j for i in data for j in i]
    for hit_0 in tqdm(search_msg):
        root_msg = hit_0
        body = {"query": {"bool": {"filter": {"match_phrase": {"q": "{}".format(root_msg)}}}}}

        questions_one = es.search(index="bot_entity_tmp_new", body=body, size=100)

        for hit_1 in questions_one['hits']['hits']:
            res = hit_1['_source']['q']
            body = {"query": {"bool": {"filter": {"match_phrase": {"q": "{}".format(res)}}}}}
            questions_two = es.search(index="bot_entity_tmp", body=body, size=100)
            out = []
            for hit_2 in questions_two['hits']['hits']:
                res_two = hit_2['_source']
                rows = {'q': str(res_two['q']).replace("\n", ""),
                        'a': str(res_two['a']).replace("\n", ""),
                        'roomId': (res_two['roomId']),
                        'tenantId': (res_two['tenantId']),
                        'lanlordId': (res_two['lanlordId']),
                        'id': str(res_two['id']).replace("\n", "")}
                out.append(rows)
            if len(out) < 1:
                continue
            df = pd.DataFrame(out)
            save_name = '{}.csv'.format(replace_symbol(out[0]['q']))
            save_dir = "/home/duyp/mayi_datasets/seed/entity"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "{}".format(save_name))

            log.info("{}".format(save_path))
            df.to_csv(save_path, index=None)


def save_question_45_to_es(index_name="question_cd_update"):
    # 批量插入
    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
        setting = {"number_of_shards": 6, "number_of_replicas": 0}
        mapping = {"timestamp": {"enabled": "true"},
                   "properties": {"logdate": {"type": "date", "format": "dd/MM/yyy HH:mm:ss"}}}

        settings = {"settings": setting, "mapping": mapping}
        es.indices.create(index=index_name, ignore=400, body=settings)
    except:
        pass

    file_dir = "antbot/datasets/city_questions_740432.csv"
    if not os.path.isfile(file_dir):
        raise FileNotFoundError("没有数据文件")
    data = pd.read_csv(file_dir).values.tolist()

    line_number = 0
    all_data = []
    source = ''
    for m in tqdm(data):
        body = {
            '_index': '{}'.format(index_name),
            '_type': 'post',
            '_id': id,
            '_source': source
        }
        all_data.append(body)
        line_number += 1
        if line_number % 10000 == 0:
            try:
                success, _ = bulk(es, all_data, index=index_name, raise_on_error=True)
                all_data = []
                log.info(
                    "==================== success :{}/{} ====================".format(line_number, len(data)))
            except Exception as e:
                log.debug("\n 存储失败！ ")