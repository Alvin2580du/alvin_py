import hashlib
from elasticsearch import Elasticsearch
es = Elasticsearch()


def curlmd5(src):
    m = hashlib.md5()
    m.update(src.encode('UTF-8'))
    return m.hexdigest()


class Entity2es(object):

    def __init__(self, entity, category, index_name='bot_entity'):
        self.entity = entity
        self.category = category
        self.enmd5 = str(curlmd5(self.entity))
        self.index_name = index_name

    def create_index(self):
        setting = {"number_of_shards": 2, "number_of_replicas": 0}
        mapping = {"timestamp": {"enabled": "true"},
                   "properties": {"logdate": {"type": "date", "format": "dd/MM/yyy HH:mm:ss"}}}
        settings = {"settings": setting, "mapping": mapping}
        if es.indices.exists(self.index_name):
            pass
        else:
            es.indices.create(index=self.index_name, ignore=400, body=settings)
            log.info("{} have create".format(self.index_name))

    def eninsert(self, new_entity=None):
        # 增
        if new_entity is None:
            self.entity = self.entity
        else:
            self.entity = new_entity
            self.enmd5 = str(curlmd5(new_entity))

        body = {'category': self.category, 'name': self.entity}
        es.index(self.index_name, doc_type="doc", id="{}".format(self.enmd5), body=body)
        log.info("{} have insert".format(self.entity))

    def update(self, new_entity=None):
        # 改
        if new_entity is None:
            self.entity = self.entity
        else:
            self.entity = new_entity

        try:
            updateBody = {"query": dict(bool={
                "must": [{"term": {"name": "{}".format(self.entity)}}]})}
            es.update_by_query(index=self.index_name, doc_type='doc', body=updateBody)
        except:
            raise Exception("No such value")

    def delete(self, new_entity=None):
        # 删
        if new_entity is None:
            self.entity = self.entity
        else:
            self.entity = new_entity
            self.enmd5 = str(curlmd5(new_entity))

        body = {'category': self.category, 'name': self.entity}

        try:
            es.delete(index=self.index_name, doc_type="doc", id=self.enmd5)
            log.info("{} have delete".format(self.entity))
        except:
            r = es.index(index=self.index_name, doc_type="doc", body=body)
            log.info("{}".format(r))
            es.delete(index=self.index_name, doc_type="doc", id=r['_id'])

    def search(self, new_entity=None):
        # 查
        if new_entity is None:
            self.entity = self.entity
        else:
            self.entity = new_entity
        out = []
        body = dict(query={"match_phrase": {"name": {"query": "{}".format(self.entity)}}})
        sort = {'name': {'order': 'asc'}}
        response = es.search(index=self.index_name, doc_type="doc", body=body, sort=sort)
        res = response['hits']['hits']

        for x in res:
            msg = x["_source"]['name']
            out.append(msg)
        if len(out) > 0:
            log.info("{} have find".format(out))
        return out