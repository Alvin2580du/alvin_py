from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
from sklearn.externals import joblib
from whoosh.fields import STORED, KEYWORD
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import RegexAnalyzer

from whoosh.index import open_dir
from whoosh.index import create_in
import os

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import parser
from urllib.request import urlopen
import jieba
from elasticsearch import Elasticsearch


jieba.load_userdict("./datasets/dictionary/jieba_dict.txt")

es = Elasticsearch()

def fun16():
    # Schema 有两个field， 标题title & 内容content
    # 当你创建index时，你只需创建一次Schema，这个Schema将与index存储在一起。
    schema = Schema(title=TEXT, content=TEXT)
    print(schema)
    """
    whoosh.fields.ID
    这个类型简单地将field的值索引为一个独立单元（这意味着，他不被分成单独的单词）。这对于文件路径、URL、时间、类别等field很有益处。

    whoosh.fields.STORED
    这个类型和文档存储在一起，但没有被索引。这个field type不可搜索。这对于你想在搜索结果中展示给用户的文档信息很有用。

    whoosh.fields.KEYWORD
    这个类型针对于空格或逗号间隔的关键词设计。可索引可搜索（部分存储）。为减少空间，不支持短语搜索。

    whoosh.fields.TEXT
    这个类型针对文档主体。存储文本及term的位置以允许短语搜索。

    whoosh.fields.NUMERIC
    这个类型专为数字设计，你可以存储整数或浮点数。

    whoosh.fields.BOOLEAN
    这个类型存储bool型

    whoosh.fields.DATETIME
    这个类型为 datetime object而设计（更多详细信息）

    whoosh.fields.NGRAM  和 whoosh.fields.NGRAMWORDS
    这些类型将fiel文本和单独的term分成N-grams（更多Indexing & Searching N-grams的信息）

    """
    schema = Schema(title=TEXT(stored=True), content=TEXT, path=ID(stored=True), tags=KEYWORD, icon=STORED)
    print(schema)

    # 用create_in函数创建index
    if not os.path.exists("index"):
        os.mkdir("index")
    ix = create_in("index", schema)
    print(ix)
    # 当你创建好索引后，你可以用open_dir打开它
    ix = open_dir("index")
    print(ix)
    # 有Index对象后可以开始添加文档。 Index对象的writer() 方法可以让你把文档加到索引上。
    writer = ix.writer()
    writer.add_document(title=u"My document", content=u"This is my document!",
                        path=u"/a", tags=u"first short", icon=u"/icons/star.png")
    writer.add_document(title=u"Second try", content=u"This is the second example.",
                        path=u"/b", tags=u"second short", icon=u"/icons/sheep.png")
    writer.add_document(title=u"Third time's the charm", content=u"Examples are many.",
                        path=u"/c", tags=u"short", icon=u"/icons/book.png")
    writer.commit()

    # Searcher 对象
    searcher = ix.searcher()
    # or
    with ix.searcher() as searcher:
        pass

    # 查询
    from whoosh.query import And, Term
    # 这个查询将会在内容field 中匹配同时包含  “apple” 和 “bear” 的文档：
    myquery = And([Term("content", u"apple"), Term("content", "bear")])

    results = searcher.search(myquery)
    print(len(results))
    print(results[0])


def fun18():
    analyzer = RegexAnalyzer(r"([\u4e00 -\u9fa5]) | (\w + (\.?\w +) * )")
    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True, analyzer=analyzer))

    ix = create_in("index", schema)
    writer = ix.writer()

    writer.add_document(title=u"First document", path=u"/a",
                        content=u"This is the first　document we’ve　added!")
    writer.add_document(title=u"Second document", path=u"/b",
                        content=u"The second one 你中文测试中文 is even more interesting!")
    writer.commit()
    searcher = ix.searcher()
    results = searcher.find("content", u"first")
    print(results)
    print(results[0])
    results = searcher.find("content", u"你")
    print(results[0])
    results = searcher.find("content", u"测试")
    print(results[0])


def load_data_and_labels(positive_data_file, negative_data_file):
    n_samples = 2000
    n_features = 1000
    n_components = 10
    n_top_words = 20

    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')

    tf = tf_vectorizer.fit_transform(x_text)
    print(tf)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    joblib.dump(lda, 'lda.model')
    # clf = joblib.load('filename.pkl')



# SDK: n57pgqs6XuOeZXF88Qsy7lFywYv99vtz


"""
列名：
    id,ownerid,alias,title,countryid,provinceid,cityid,districtid,streetid,leasetype,
    type,mindays,maxdays,payrule,offlinededuct,refunddays,checkintime,checkouttime,
    guestnum,roomnum,privatetoiletnum,publictoiletnum,area,stock,mainimageid,dayprice,
    weekprice,monthprice,weekendtype,weekendprice,createtime,operid,lastupdatetime,
    firstcommittime,firstonlinetime,onlinetime,offlinetime,state,lastaudittime,property,
    shophours,latlng,address,displayaddr,lastauditremark,bedtype,facilities,images,detail,
    fastpaystate,fastpayforbidendtime,fastpayforbidreason,lastaudittype,roomrank,parlor,
    familylodgestate,realtimestate,realnamestate,onedayrefundstate,gdlatlng,sharelodgestate,
    detailaddr,specialdiscount,boutique,linenstate,elongflag,cellar,cookhouse,balcony,service,
    moreservice,depositstate,deposit,checktrue,checktruetime,checktrueuser,nextdoorid,publishfrom,
    housemark,sheetreplace,publishstep,nocontinue,locid,zmlodge,lodgelevel,lodgecomprelevel,
    housenumber,product,platform,smalldistrict,currency
"""


def request1(appkey, m='GET'):
    url = "http://op.juhe.cn/189/bus/busline"
    params = {"key": appkey,
              "dtype": "",
              "city": "75",
              "bus": "1"
              }
    params = parse.urlencode(params)
    if m == "GET":
        f = urlopen("%s?%s" % (url, params))
    else:
        f = urlopen(url, params)

    content = f.read()
    res = json.loads(content)
    if res:
        error_code = res["error_code"]
        if error_code == 0:
            print(res["result"])
        else:
            print("%s:%s" % (res["error_code"], res["reason"]))
    else:
        print("request api error")


def fun(landlordid=850266459, roomid=851776620):
    res = get_info_from_lodgeunit_by_ownerid(landlordid, city_id=45)
    for x in res:
        if roomid == x['id']:
            out = []
            out_remark = []
            out_more = []
            facilities = x['facilities']

            fac = json.loads(facilities)

            for d in fac:
                rows = OrderedDict()
                d = dict(d)
                if len(d) == 2:
                    name = d['name']
                    price = d['price']
                    rows[name] = price
                if len(d) == 4:
                    name = d['name']
                    price = d['price']
                    rows[name] = price
                    rows['remark'] = d['remark']
                    rows['more'] = d['more']
                print(rows)


def all_case():
    """
    TYPE, S, P, REL, NUM, OP:AND, OP:OR
    """
    out = []
    s = "TYPE,S,P,REL,NUM,OP:AND,OP:OR"

    s_sp = s.split(",")
    for i in range(2, 10):
        for x in combinations(s_sp, i):
            if "TYPE" in x:
                for y in permutations(x, len(x)):
                    y = list(y)
                    print(y)
                    if y[0] == "TYPE" and "OP" not in y[1] and "OP" not in y[-1] and "REL" not in y[1] and "REL" not in \
                            y[-1]:
                        if "OP:AND" and "OP:OR" in y:
                            continue
                        # if "REL" and " OP:OR" in y:
                        #     continue
                        # if "REL" and " OP:AND" in y:
                        #     continue
                        if "P" and "S" in y:
                            print("1")
                            if y.index("P") > y.index("S"):
                                continue
                        else:
                            out.append(y)
    out_final = []
    for x in out:
        res = "]]::[[".join(x)
        res = "[[{}]]".format(res)
        out_final.append(res)

    out_final_1 = []
    for x in out_final:
        out_final_1.append(x)
        if "OP:AND" in x:
            x_re = x.replace("OP:AND", "OP:OR")
            out_final_1.append(x_re)

    fw = open("combinations.txt", 'w')
    for i in out_final_1:
        fw.writelines(i + "\n")


def list_dir(path="./datasets/seed/results/cd"):
    import os
    from collections import Counter
    import jieba

    names = []
    for file in os.listdir(path):
        names.append(file.split(".")[0])

    for name in names:
        print(name)


def q_cixin(path="./datasets/question_45/keywords/key_question_1"):
    with open(path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = posseg.cut(line)
            for x in res:
                flag = x.flag
                word = x.word
                print(flag, word)


def fun1():
    question_list = ['远', '简直', '来', '如果', '哪里', '近', '有不', '请问', '麻烦发下',
                     '多少', '有啥', '方便', '哪边儿', '去', '还有', '多大', '怎的', '离', '竟然',
                     '是否', '有吗', '何人', '何时', '哪儿', '在么', '哪一个',
                     '吧', '呢', '还是', '到', '什么', '怎么', '可以', '免费不', '啊', '多长', '有没', '谁的',
                     '为啥', '几', '收费不', '能不能', '请告知', '没有', '怎样', '别的', '如何', '么',
                     '吗', '哪个', '哪', '几时', '在不', '是不是', '为何', '多久', '怎么着', '行不行',
                     '居然', '嘛', '的吗', '租吗', '订吗', '车', '机']

    print(len(question_list))
    ou = {}
    for x in question_list:
        body = dict(query={"match": {"msg": {"query": "{}".format(x)}}})
        ret = es.search(index="question_cd_md5", body=body, size=2000)
        total = ret['hits']['total']
        if total < 10000:
            question_list.remove(x)
            print(x, total)

    print(question_list)


def fun2():
    chatbot = ['chatbot1', 'chatbot2', 'chatbot3', 'chatbot4']
    md5 = ''
    for cb in chatbot:
        question = ''
        answer = ''
        roomid = ''
        body = {
            '_index': '{}'.format(cb),
            '_type': 'doc',
            '_id': '{}'.format(md5),
            '_source': {
                'msg': question,
                "{}".format(roomid): answer}
        }


def fun3():
    import os
    import Levenshtein

    files = os.listdir("./datasets/seed/results/cd/")
    file_list = []
    for file in files:
        print(file.split(".")[0])

        file_list.append(file)

    length = len(file_list)
    out = []
    for i in range(length):
        rows = []
        for j in range(i + 1, length):
            x1, x2 = file_list[i], file_list[j]
            similar = Levenshtein.ratio(x1, x2)
            if similar > 0.85:
                if x1 not in rows:
                    rows.append(x1)
                if x2 not in rows:
                    rows.append(x2)
            else:
                out.append(rows)
                rows = []
        if len(rows) > 0:
            print(rows)

    for similar_file in out:
        print(similar_file)


def fun4():
    from pyduyp.utils.fileops import sort_file_by_dict

    file_name = "./datasets/seed/results/"
    sort_file_by_dict(file_name, "quesion_list_all.csv", "q_all.csv")


def fun5():
    out = []

    with open("./datasets/facilities/prop.txt", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split(",")[-1]
            if line_sp not in out:
                out.append(line_sp.strip())
    print(out)


def file_to_line(filename):
    data = pd.read_csv(filename).values.tolist()
    data_faci = [j for i in data for j in i]
    return "".join(str(data_faci))


def fun6():
    data_faci = pd.read_csv("./datasets/question_45/keywords/facilities_cd").values.tolist()
    data_faci = [j for i in data_faci for j in i]

    files = glob(os.path.join("/home/duyp/mayi_datasets/cd", "*.csv"))
    out_file = []
    for f in tqdm(files):
        f_name = f.split("/")[-1]
        res = file_to_line(f)
        key = is_inline(res, data_faci)
        if key:
            out_file.append(f_name)
    df = pd.DataFrame(out_file)
    df.to_csv("./datasets/q_list_final.csv", index=None)


def fun7():
    fw = open("./datasets/question_list_final.csv", 'w')

    with open("./datasets/q_list_final.csv", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split(".")[0]
            fw.writelines(line_sp.strip() + "\n")


def fun8():
    from shutil import copy

    path = "/home/duyp/mayi_datasets/cd"
    path1 = "/home/duyp/mayi_datasets/cd_faci"
    if not os.path.exists(path1):
        os.makedirs(path1)
    files = glob(os.path.join(path, "*.csv"))
    files1 = []
    with open("./datasets/q_list_final.csv", 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            file_name = os.path.join(path, line)
            files1.append(file_name.strip())

    res = [i for i in files if i in files1]

    for x in res:
        copy(x, os.path.join(path1, x.split("/")[-1]))


def fun9():
    da = pd.read_csv("./datasets/question_45/keywords/facilities_cd").values.tolist()
    facilites = [j for i in da for j in i]

    path = "/home/duyp/mayi_datasets/cd_faci"
    out = []
    files = os.listdir(path)
    line_number = 0
    for f in files:
        fname = os.path.join(path, f)
        data = pd.read_csv(fname).values.tolist()
        for x in data:
            line_number += 1
            msg = x[0]
            if is_inline(msg, facilites):
                out.append(x)
    df = pd.DataFrame(out)
    df.to_csv("./datasets/question_45/message_facilities.csv", index=None)


def fun10():
    n_features = 10000
    n_components = 10

    doc = open("./datasets/cd_new.csv", 'r')
    lines = doc.readlines()

    # vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
    #                              min_df=2, stop_words='english',
    #                              use_idf=use_idf)
    hasher = HashingVectorizer(n_features=n_features,
                               stop_words='english', alternate_sign=False,
                               norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())

    X = vectorizer.fit_transform(lines)
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    # init = k-means++, random
    km = MiniBatchKMeans(n_clusters=n_components, init="random", max_iter=1000, batch_size=10000,
                         verbose=0, compute_labels=True, random_state=None, tol=0.0,
                         max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)

    km.fit(X)
    joblib.dump(km, 'km.model')


def km_test():
    n_features = 10000
    hasher = HashingVectorizer(n_features=n_features,
                               stop_words='english', alternate_sign=False,
                               norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    fr = open("./datasets/cd.csv", 'r')
    lines = fr.readlines()
    print(len(lines))
    x = vectorizer.fit_transform(lines)

    km = joblib.load("km.model")
    y_kmeans = km.fit_predict(x)

    data = pd.Series(y_kmeans)
    data.to_csv("antbot/datasets/results/cd_predict.csv", index=None, header=None)


def is_inline(line, lis):
    outputs = False
    for x in lis:
        if x in line:
            outputs = True
    return outputs


def all_mes():
    room_facilities = []
    with open("./datasets/entity/house.csv", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            room_facilities.append(line.strip())
    fw = open("./datasets/results/not_facilities.txt", 'a+')
    with open("./datasets/question_45/cd_new.txt", 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            line_sp = line.replace(" ", "")
            if not is_inline(line_sp, lis=room_facilities):
                print(line)
                # fw.writelines(line)


from collections import Counter


def gongjiao_spyder():
    import urllib.request
    import re

    urls = []
    with open("./datasets/urls", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split('"')[1]
            urls.append(line_sp)

    for x in tqdm(urls):
        fw = open("/home/duyp/mayi_datasets/gongjiao/{}.txt".format(x.split("/")[-1]), 'w')
        res = urllib.request.urlopen(x).read().decode("utf-8")
        fw.writelines(res)


def help_fun(file):
    out = ""
    with open(file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            out += line.replace("\n", "").replace(" ", "")
    return out


def GetMiddleStr(content, startStr, endStr):
    startIndex = content.index(startStr)
    if startIndex >= 0:
        startIndex += len(startStr)
    endIndex = content.index(endStr)
    return content[startIndex:endIndex]


def find_gongjiaozhan():
    root = '/home/duyp/mayi_datasets/gongjiao'
    import re
    fw = open("./datasets/dictionary/gj.txt", 'w')
    out = []
    for file in os.listdir(root):
        file_name = os.path.join(root, file)

        res = help_fun(file_name)
        reg = '{}.+?{}'.format("返程路线", "公交线路详情页面")
        gj = re.compile(reg).search(res)
        if gj:
            gj = gj.group(0)

            gj_list = re.compile("[\u4E00-\u9FA5]{2,10}").findall(gj)
            for x in gj_list:
                if x == "返程路线":
                    continue
                if x == "公交线路详情页面":
                    continue
                out.append(x)

    for x in out:
        fw.writelines(x + "\n")


def jieba_cut(line):
    res = jieba.lcut(replace_symbol(delete_imjo(line.replace('"', ""))))
    return " ".join(res)


def cut_file():
    import jieba
    jieba.load_userdict("./datasets/dictionary/jieba_dict.txt")

    data_source = pd.read_csv("./datasets/question_45/cd.csv").values.tolist()
    da = [j for i in data_source for j in i]
    out = []
    for x in tqdm(da):
        res = jieba_cut(x)
        out.append(res)
    df = pd.DataFrame(out)
    df.to_csv("./datasets/question_45/results/cd_cut.csv", index=None)


def jiaoji(l1, l2):
    out = False
    res = [i for i in l1 if i in l2]
    if len(res) > 0:
        out = True
    return out


def traffic_q():
    data = pd.read_csv("./datasets/question_45/results/cd_cut.csv", header=None).values.tolist()
    data = [j for i in data for j in i]

    l2 = pd.read_csv("./datasets/question_45/keywords/keywords.txt").values.tolist()
    l2 = [j for i in l2 for j in i]

    out = []
    for x in tqdm(data):
        x_sp = x.split(" ")
        if jiaoji(x_sp, l2):
            out.append(x)
    df = pd.DataFrame(out)
    df.to_csv("./datasets/question_45/results/traffic.csv", index=None)


def plot_word_cloud():
    from wordcloud import WordCloud, ImageColorGenerator

    text = open("./datasets/question_45/results/traffic.csv", 'r').read()
    wc = WordCloud(background_color="white", width=1000, height=800,
                   max_font_size=40,
                   random_state=42,
                   max_words=300,
                   font_path='/home/duyp/fonts/msyh.ttf')
    wc.generate(text)
    wc.to_file(os.path.join("./datasets/question_45/results", "traffic.png"))


import re


def split_question(questions):
    out_list = []
    if isinstance(questions, str):
        if "?" in questions or "？" in questions:
            res = re.split("[\?\？\么\吗\呢\吧\嘛]", questions)
            if res:
                for x in res:
                    if len(x) > 0:
                        out = "{}吗".format(x)
                        out_list.append(out)
        else:
            return questions
    if len(out_list) == 1:
        return out_list[0]
    else:
        return out_list

import requests
import hashlib
from tqdm import tqdm
from py2neo import Node
from py2neo import Relationship
import re
import json
import os
import pandas as pd
from glob import glob
from time import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from gensim import corpora, models, similarities

from pyduyp.logger.log import log
from pyduyp.utils.string import is_question

es = Elasticsearch()

root = os.path.dirname(os.path.realpath(__file__))
log.debug("root : {}".format(root))


def create_report(data):
    url = 'http://localhost:9200/reports/report/'

    headers = {'content-type': 'application/json'}

    r = requests.post(url, data=json.dumps(data), headers=headers)
    res = json.loads(r.text)
    return res


def get_all_datasets():
    search_obj = {
        "size": 0,
        "aggs": {
            "langs": {
                "terms": {"field": "chart.data.url.keyword"}
            }
        }
    }

    url = 'http://localhost:9200/_search'
    headers = {'content-type': 'application/json'}
    r = requests.post(url, data=json.dumps(search_obj), headers=headers)

    return r.text


def get_all_charts():
    search_obj = {
        "size": 0,
        "aggs": {
            "chart_types": {
                "terms": {
                    "field": "chart.type.keyword"
                }
            },
            "chart_ids": {
                "terms": {
                    "field": "chart._id"
                }
            }
        }
    }

    url = 'http://localhost:9200/_search'
    headers = {'content-type': 'application/json'}
    r = requests.post(url, data=json.dumps(search_obj), headers=headers)
    return r.text


def get_reports_by_tag(tag):
    search_obj = {
        "query": {
            "query_string": {
                "query": tag,
                "fields": ['tags']
            }
        }
    }

    url = 'http://localhost:9200/_search'
    headers = {'content-type': 'application/json'}
    r = requests.post(url, data=json.dumps(search_obj), headers=headers)
    return r.text


def get_all_reports():
    url = 'http://localhost:9200/reports/report/_search'
    headers = {'content-type': 'application/json'}
    r = requests.get(url, headers=headers)
    return r.text


def update_mapping(obj, field):
    map_obj = {
        "properties": {
            field: {
                "type": "text",
                "fielddata": True
            }
        }
    }

    url = 'http://localhost:9200/reports/_mapping/my_type?update_all_types'
    headers = {'content-type': 'application/json'}
    r = requests.post(url, data=json.dumps(map_obj), headers=headers)
    return r.text


def split_question(data_name='question_list.txt'):
    data_dir = os.path.join(root, "datasets")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = os.path.join(data_dir, data_name)

    results = []
    line_number = 0
    with open(data_dir, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if len(line.split(",")) == 2:
                q = line.split(',')[1]
                results.append(q)
                line_number += 1
                if line_number % 100000 == 0 or line_number == len(lines):
                    df = pd.DataFrame(results)
                    save_path = os.path.join(root, "datasets")
                    save_path_name = os.path.join(save_path, "question_list_{}.json".format(line_number))
                    log.debug(" file have saved: {}".format(save_path_name))
                    df.to_json(save_path_name)
                    results = []
            else:
                continue


def string2dict(inputs):
    inputsreplace = inputs.replace("'", "").replace("{", "").replace("}", "").replace('"', "")
    inputs2split = inputsreplace.split(",")
    rows = {}
    for x in inputs2split:
        if len(x.split(":")) == 2:
            x1, x2 = x.split(":")[0], x.split(":")[1]
            rows[x1] = x2
    return rows


def read_json2es(index_name, data_path, data_name=None):
    #  保存到es
    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
    except:
        pass

    es.indices.create(index_name)
    if data_name:
        data_real_name = data_name.split('.')[0]
        data_read_name = os.path.join(root, data_path, data_name)
        json_rows = open(data_read_name, 'r')
        data = json.load(json_rows)
        for name, values in data.items():
            # log.debug("name: {}, values: {}".format(name, values))
            es.index(index_name, doc_type="post", id="{}".format(data_real_name),
                     body={"{}".format(name): "{}".format(values)})
    else:
        file_base_dir = os.path.join(root, data_path)
        files = glob(os.path.join(file_base_dir, "*.json"))
        for file in files:
            file_real_name = os.path.basename(file).split('.')[0]
            json_rows = open(file, 'r')
            data = json.load(json_rows)
            for name, values in data.items():
                # log.debug("name: {}, values: {}".format(name, values))
                es.index(index_name, doc_type="post", id="{}".format(file_real_name),
                         body={"{}".format(name): "{}".format(values)})
            log.debug("file have save to elasticsearch :{}".format(file_real_name))


def search_content(index_name, id):
    # 查询
    result = es.get(index_name, doc_type="post", id=id)
    log.debug("results keys : {}".format(result.keys()))
    for k in result['_source']:
        index = k
        index_values = result['_source'][k]
        index_values2dict = string2dict(index_values)
        for line, values in index_values2dict.items():
            pass
            log.info("line: {}, message: {}".format(line, values))


def delete_index(index_name, id=None, method="one"):
    # 删除
    if method == "one":
        es.indices.delete(index_name)
    else:
        es.delete(index_name, doc_type="test_type", id=id)


def question_300w2es(data_name, index_name="question_300w"):
    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
    except:
        pass
    es.indices.create(index=index_name)
    log.info("======= index build success ======")
    # 读入数据
    data_dir = os.path.join(root, "datasets")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_dir = os.path.join(data_dir, data_name)
    all_data = []
    line_number = 0
    with open(data_dir, 'r') as fr:
        lines = fr.readlines()
        length = len(lines)
        for line in lines:
            if len(line.split(",")) == 2:
                q = line.split(',')[1]
                all_data.append({
                    '_index': 'question_300w',
                    '_type': 'question_300w_type',
                    '_id': '{}'.format(line_number),
                    '_source': {
                        'title': q
                    }
                })
                if line_number % 10000 == 0 or line_number == len(lines):
                    success, _ = bulk(es, all_data, index='question_300w', raise_on_error=True)
                    all_data = []
                    log.info("==================== success :{}/{} ====================".format(success, length))
                line_number += 1


def search_by_keywords_by_title(keywords):
    # 查
    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "title": "{}".format(keywords)
                    }
                ]
            }
        }
    }
    response = es.search(index="question_300w", doc_type='question_300w_type', body=body)

    for row in response['hits']['hits']:
        for k, v in row.items():
            print(k, v)
        print("=========================")


def question_300w2es_by_question(index_name="question_300w"):
    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
    except:
        pass
    es.indices.create(index=index_name)
    log.info("======= index build success ======")
    # 读入数据
    data_dir = os.path.join(root, "datasets")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    all_data = []
    files = glob(os.path.join(data_dir, "*.csv"))
    log.info("Total files number : {}".format(len(files)))
    for file in files:
        line_number = 0

        file_name = os.path.basename(file).split(".")[0]
        with open(file, 'r') as fr:
            lines = fr.readlines()
            length = len(lines)
            for line in lines:
                line_number += 1

                if len(line.split(",")) == 2:
                    q = line.split(',')[1]
                    print(q)
                    all_data.append({
                        '_index': 'question_300w',
                        '_type': 'question_300w_{}'.format(file_name),
                        '_id': '{}'.format(line_number),
                        '_source': {
                            'title': q
                        }
                    })
                    if line_number % 100 == 0 or line_number == len(lines):
                        success, _ = bulk(es, all_data, index=index_name, raise_on_error=True)
                        all_data = []

                    if line_number % 20000 == 0:
                        log.info("==================== success :{}/{} ====================".format(line_number, length))


def read_types():
    types = []
    with open("./datasets/intention_word_list_过来_筛选.txt", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if len(line.split("_")) == 2:
                line_sp = line.split(",")[1].split("_")[0]
                types.append(line_sp.strip("\n"))
    df = pd.DataFrame(types)
    df.columns = ['q_types']
    df.to_csv("./datasets/question_types.txt", index=None)
    return types


def classification_helper(inputs):
    if isinstance(inputs, str):
        ds = "[\\n\ \-\"\+\……\（\）\_\、\。\？\～\；\~\!\@\#\$\^\&\(\)\=\|\{\}\'\:\;\,\，\[\]\.\<\>\/\?\~\！\#\\\&\*\%]"
        inputs2sub = re.sub(ds, "", inputs)
        types_data = pd.read_table("./datasets/question_types.txt").values.tolist()
        types = [j for i in types_data for j in i]
        log.debug("types: {}".format(types))

        for x in types:
            log.debug("{}".format(x))
            if x in inputs2sub:
                log.debug("{}".format(x))
                return "{}+{}".format(inputs2sub, x)
            else:
                return " "
    else:
        return " "


def clf2es(index_name='clf_list'):
    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
    except:
        pass
    data = pd.read_table("./datasets/search_clf_list.txt", sep=',', usecols=[2])
    verbol_list = data.values

    counter = 1
    for x in verbol_list:
        x2str = str(x).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        s2str2list = x2str.split(",")
        rows = []
        for i in s2str2list:
            rows.append(i.strip())

        rows2dict = {"{}".format(rows[0]): rows}
        encode_json = json.dumps(rows2dict)
        decode_json = json.loads(encode_json)

        for name, values in decode_json.items():
            body = {
                "text": {
                    "{}".format(name): "{}".format(values)
                }
            }

            es.index(index_name, doc_type="post", id="{}".format(counter), body=body)

        counter += 1


def hanLP():
    from jpype import startJVM, getDefaultJVMPath, JClass, shutdownJVM

    startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/duyp/HanLp/hanlp-1.5.2.jar:/home/duyp/HanLp", "-Xms1g",
             "-Xmx1g")
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')

    # 中文分词
    seg = HanLP.segment('你好，欢迎在Python中调用HanLP的API')
    print(seg, type(seg))

    testCases = ["商品和服务",
                 "结婚的和尚未结婚的确实在干扰分词啊",
                 "买水果然后来世博园最后去世博会",
                 "中国的首都是北京",
                 "欢迎新老师生前来就餐",
                 "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
                 "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。"]
    for sentence in testCases:
        segs = HanLP.segment(sentence)

    # 命名实体识别与词性标注
    x1 = NLPTokenizer.segment('中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程')
    x12string = x1.toString()
    print(x12string)
    x12string2split = x12string.replace("[", "").replace("]", "").split(",")
    for i in x12string2split:
        if i.split("/")[1] == "nt" or i.split("/")[1] == "nnt":
            print(i)

    # 关键词提取
    document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
               "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
               "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
               "严格地进行水资源论证和取水许可的批准。"
    x2 = HanLP.extractKeyword(document, 2)

    # 自动摘要
    x3 = HanLP.extractSummary(document, 3)
    # 依存句法分析
    x4 = HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。")
    shutdownJVM()


def hanLP_for_enti():
    # 命名实体识别与词性标注

    from jpype import startJVM, getDefaultJVMPath, JClass

    startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/duyp/HanLp/hanlp-1.5.2.jar:/home/duyp/HanLp", "-Xms1g",
             "-Xmx1g")

    NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')

    testCases = ["商品和服务迎宾宾馆天安门都江堰",
                 "结婚的和尚未结婚的确实在干扰分词啊",
                 "买水果然后来世博园最后去世博会",
                 "中国的首都是北京",
                 "欢迎新老师生前来就餐",
                 "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
                 "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。",
                 '中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程'
                 ]

    enties = {"ntu": "大学", "nt": "机构团体名", "ntc": "公司名", "ntcb": "银行",
              "ntcf": "工厂", "ntch": "酒店宾馆", "nth": "医院", "nto": "政府机构",
              "nts": "中小学"}

    line = 1
    res = []
    for sentence in testCases:
        rows = []
        x1 = NLPTokenizer.segment(sentence)
        x12string = x1.toString()
        x12string2split = x12string.replace("[", "").replace("]", "").split(",")
        for i in x12string2split:
            if i.split("/")[1] in enties.keys():
                rows.append(i)
        rows2dict = {line: rows}

        if len(rows) > 0:
            res.append(rows2dict)
    return res


def curlmd5(src):
    m = hashlib.md5()
    m.update(src.encode('UTF-8'))
    return m.hexdigest()


def is_in_string(inputs):
    outputs = inputs
    for t in types:
        if t not in inputs:
            outputs = "{}".format(outputs)
        else:
            outputs = "{} {}".format(outputs, t)
    return outputs


def question_classification():
    question_dir = os.path.join(root, "datasets", "question_list.txt")
    data = pd.read_table(question_dir, sep=',', skiprows=1, usecols=[1], header=None)
    log.info("data length : {}".format(data.shape))
    data.columns = ['message']
    t0 = time()
    data_label = data['message'].apply(classification_helper)
    log.info("time cost: {}".format(time() - t0))
    data_label2df = pd.DataFrame(data_label)
    data_label2df.columns = ["message"]
    out_df = (data_label2df["message"].isnull()) | (data_label2df["message"].apply(lambda x: str(x).isspace()))
    out_df_res = data_label2df[~out_df]
    log.info("out_df_res length : {}".format(out_df_res.shape))
    save_dir = os.path.join(root, "datasets", "question_classification.csv")
    out_df_res.to_csv(save_dir, sep=" ", index=None)


def nltk_learn():
    import nltk
    import jieba
    text = '命名实体识别北京'
    tokens_jieba = jieba.lcut(text)
    tagged_jieab = nltk.pos_tag(tokens_jieba)
    entities_jieba = nltk.chunk.ne_chunk(tagged_jieab)
    print(entities_jieba)
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)  # 命名实体识别
    print(entities)


# 命名实体识别
def ner(words, postags):
    from pyltp import SentenceSplitter
    from pyltp import Segmentor
    from pyltp import Postagger
    from pyltp import NamedEntityRecognizer

    recognizer = NamedEntityRecognizer()
    recognizer.load('E:\\ltp-data-v3.3.1\\ltp_data\\ner.model')
    netags = recognizer.recognize(words, postags)
    for word, ntag in zip(words, netags):
        print(word + '/' + ntag)

    recognizer.release()  # 释放模型
    nerttags = list(netags)
    return nerttags


def similar_message2es(index_name='similarmessage'):
    from jieba import analyse

    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
    except:
        pass

    counter = 1
    s2str2list = []
    with open("./datasets/similar_test", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            s2str2list.append(line.strip("\n"))

    all_test = "".join(s2str2list)
    keywords = analyse.extract_tags(all_test, topK=1)

    rows2dict = {"{}_{}".format(keywords[0], keywords[1]): s2str2list}
    encode_json = json.dumps(rows2dict)
    decode_json = json.loads(encode_json)

    for name, values in decode_json.items():
        body = {
            "text": {
                "{}".format(name): "{}".format(values)
            }
        }

        es.index(index_name, doc_type="post", id="{}".format(counter), body=body)

    counter += 1


def extract_keywords():
    from jieba import analyse
    import jieba
    jieba.load_userdict("./datasets/userdict.txt")
    s2str2list = []
    with open("./datasets/similar_test", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_cut = jieba.lcut(line.strip())
            print(line_cut)


def search_by_keywords(size, keywords='有房_今天_做饭', index_name='question_444w', is_include_must_not=True):
    # 根据关键词去es中查数据,返回消息列表
    if is_include_must_not:
        k1, k2, k3 = keywords.split("_")
        body = {
            "query": {
                "bool": {
                    "must": {"match": {"title": "{}".format(k1)}},
                    "must_not": {"match": {"title": "{}".format(k3)}},
                    "should": {"match": {"title": "{}".format(k2)}},
                }
            }
        }
    else:
        body = {
            "query": {
                "bool": {
                    "must": {"match": {"title": "{}".format(keywords)}}
                }
            }
        }
    sort = {'title': {'order': 'asc'}}

    response = es.search(index=index_name, doc_type="post", body=body, size=size, sort=sort)

    res = response['hits']['hits']
    msg_list = []
    for x in res:
        msg = x["_source"]['title']
        msg_list.append(msg)
    return msg_list


def test_es_search():
    res = []
    with open("./datasets/search_clf_list.txt", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split(",")
            for x in line_sp:
                x_replace = x.replace("'", "").replace("[", "").replace("]", "").replace('"', "").replace("\n", "")
                if not x_replace.isdigit():
                    res.append(x_replace)

    save_message = {}
    for x in res:
        search_results = search_by_keywords(size=200, keywords='{}'.format(x), index_name='question_444w',
                                            is_include_must_not=False)
        save_message[x] = search_results

    for k, v in save_message.items():
        save_path = "./datasets/questions/{}.csv".format(k)
        df = pd.DataFrame(v)
        df.to_csv(save_path, index=None)
        log.info("save success ")


def fun2():
    from pyduyp.utils.fileops import sort_file_by_dict
    import os
    files = os.listdir("./datasets/questions/question_new")
    for file in files:
        sort_file_by_dict(data_dir='./datasets/questions/question_new',
                          input_filename=file, output_filename="{}_sorted.csv".format(file.split(".")[0]))


def get_all_userpaires():
    from pyduyp.utils.fileops import dump, checkexist, load
    from pyduyp.datasources.redis_db import redis
    step_path = 'antbot/datasets/question_step.txt'
    session_userpairs_total = 's:msg:userpairs:total'
    save_path = 'antbot/datasets/{}.txt'.format(session_userpairs_total)

    if not checkexist(save_path):
        userpairs = redis.instance().smembers(session_userpairs_total)
        dump(save_path, userpairs)
    else:
        userpairs = load(save_path)
    userpairs = list(userpairs)
    if checkexist(step_path):
        loaditinfo = load(step_path)
        if loaditinfo is not None:
            userpairs = userpairs[loaditinfo:]
    df = pd.DataFrame(list(set(userpairs)))
    df.to_csv("antbot/datasets/userpairs_all.csv", index=None, header=None)


def get_all_message_by_userpairs(userpairs_file):
    from pyduyp.datasources.immsg import get_tenant_message_by_tenant
    from pyduyp.utils.string import removespecialchar

    with open(userpairs_file, 'r') as fr:
        lines = fr.readlines()
        results = []
        i = 0
        for line in tqdm(lines):
            from_id = line.replace("b", "").replace("'", "").split(":")[0]
            ret = get_tenant_message_by_tenant(from_id)
            if ret is None or len(ret) == 0:
                continue
            for row in ret:
                row = dict(row.items())
                msg = removespecialchar(row['msg'])
                if len(msg) == 0:
                    continue
                results.append(msg)
                i += 1
                if i % 10000 == 0:
                    log.info("==========================={} ===================".format(i))

                if i % 500000 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv("./datasets/mesg_{}.csv".format(i), index=None, header=None)
                    log.info(" =====================Success =====================")
                    results = []


def centry_by_landlord(landlord_id="12345111"):
    """
    :param landlord_id:
    :return:
        landlord_id wifi size title tv ... room_id
        1   2   3   4   ....    n

    """
    landlord = Node("Landlord", name="{}".format(landlord_id))
    landlord['name'] = "房东"
    landlord['mobile'] = "130191999999"

    house1 = Node("House", name="123456")
    house1['wifi'] = 1
    house1['size'] = 30
    house1['title'] = "完美实拍好房子"
    house1['tv'] = 1
    house1['location'] = "北京市海淀区上地三街金禹嘉华大厦1001"

    house2 = Node("House", name="123457")
    house2['wifi'] = 1
    house2['size'] = 30
    house2['title'] = "完美实拍好房子"
    house2['tv'] = 1
    house2['location'] = "北京市海淀区上地三街金禹嘉华大厦1002"

    ab1 = Relationship(landlord, "OWN", house1)
    ab2 = Relationship(landlord, "OWN", house2)

    print(landlord, house1, ab1)
    print(landlord, house2, ab2)


def get_facilities():
    from pyduyp.datasources.lodgeunit import get_facilities_from_lodgeunit, get_facilities_from_lodgeunit_test
    res = get_facilities_from_lodgeunit_test()
    out = {}
    for m in res:

        facilities, room_id, ownerid = m['facilities'], m['id'], m['ownerid']
        facilities_replace = facilities.replace('"', "")
        name = re.compile("name:[\w\u2E80-\u9FFF]+,price:[+-]\d+").findall(facilities_replace)
        res = {}
        for x in name:
            k = x.split(",")[0].split(":")[-1]
            v = x.split(",")[-1].split(":")[-1]
            res[k] = v
        out[room_id] = res
    return out


def send2tenant_message(roomid):
    res = get_facilities_by_id(roomid)
    out = "您好! 下面是小葱知道的房源基本情况,请陛下过目:\n"
    for k, v in res.items():
        if v == '0':
            price = '免费'
        elif v == '-1':
            price = "否"
        elif v == '1':
            price = "收费"
        else:
            raise Exception("错误")
        out += "{},{} \n".format(k, price)

    return out


#
# res = send2tenant_message_v1(roomid='850375633', keyword='电视')
# print(res)

def all_facilities_names2txt(save_name='facilities.txt'):
    root = 'antbot/datasets/facilities'

    from pyduyp.datasources.lodgeunit import get_facilities_from_lodgeunit
    res = get_facilities_from_lodgeunit()
    out = []
    for m in res:
        facilities, room_id, ownerid = m['facilities'], m['id'], m['ownerid']
        p = re.compile("\d*[\u2E80-\u9FFF]+").findall(facilities)
        for x in p:
            if x not in out:
                out.append(x)
    save_dir = os.path.join(root, save_name)
    fw = open(save_dir, 'w')
    for x in out:
        fw.writelines(x + "\n")


def delete_common_message(file_name):
    import Levenshtein
    out = []
    with open(file_name, 'r') as fr:
        lines = fr.readlines()
        length = len(lines)
        for i in range(length):
            for j in range(i, length):
                simi = Levenshtein.ratio(lines[i], lines[j])
                if simi < 0.9 and lines[i] not in out:
                    out.append(lines[i])
    df = pd.DataFrame(out)
    df.to_csv("./datasets/out.csv", index=None, header=None)


def get_facilities_by_id(roomid):
    #
    from pyduyp.datasources.lodgeunit import get_facilities_from_lodgeunit_by_id
    res = get_facilities_from_lodgeunit_by_id(roomid)
    for m in res:
        facilities, room_id, ownerid = m['facilities'], m['id'], m['ownerid']
        fac = json.loads(facilities)
        room_properties = {'room_id': room_id}

        for d in fac:
            d = dict(d)
            name = d['name']
            price = d['price']
            room_properties[name] = price

            if d.get('remark') is not None:
                remark = d['remark']
                room_properties["remark"] = remark

            if d.get('more') is not None:
                more = d['more']
                room_properties["more"] = more
        return room_properties


def all_facilities_names2txt_v1(save_name='facilities.csv', property_name='prop.txt'):
    from pyduyp.datasources.lodgeunit import get_facilities_from_lodgeunit
    # 获取所有房源的设备名字
    root = 'antbot/datasets/facilities'
    if not os.path.exists(root):
        os.makedirs(root)
    res = get_facilities_from_lodgeunit()
    out = []
    start = 1
    limit = 10
    rooms = []
    total = len(res)
    for m in res:
        facilities, room_id, ownerid = m['facilities'], m['id'], m['ownerid']
        print(facilities)
        fac = json.loads(facilities)
        room_properties = {'room_id': room_id}

        for d in fac:
            d = dict(d)
            name = d['name']
            price = d['price']
            remark = ''
            if d.get('remark') is not None:
                remark = d['remark']
            more = ''
            if d.get('more') is not None:
                more = d['more']
            log.debug("{} {} {} {}".format(name, price, remark, more))
            if name not in out:
                room_properties[name] = price
                out.append(name)
            if remark is not None and remark not in out and remark != '':
                room_properties[remark] = price
                out.append(remark)
            if more is not None and more not in out and more != '':
                room_properties[more] = price
                out.append(more)
        rooms.append(room_properties)
        start += 1
        # if start > limit:
        #     break
        log.debug("get process {} / {}".format(start, total))

    # save_dir = os.path.join(root, save_name)
    # room_pd = pd.DataFrame(rooms)
    # room_pd.to_csv(save_dir)
    #
    # prop_dir = os.path.join(root, property_name)
    # prop_pd = pd.DataFrame(out)
    # prop_pd.to_csv(prop_dir)


def send2tenant_message_v1(roomid, keyword):
    # keyword为从房客提问中提取的实体信息并转换为标准关键词
    """
    公共配套设施:
    ['无线WIFI', '电梯', '停车位', '空调', '暖气', '冰箱', '洗衣机', '电视', '拖鞋', '淋浴', '24小时热水',
     '沐浴露洗发水', '牙具香皂', '无障碍设施', '欢迎小孩', '欢迎老人', '收起有线网络', '电脑', '饮水机', '浴缸',
      '毛巾浴巾', '保安', '对讲系统', '早餐', '可以吸烟', '允许聚会', '可养宠物']

    :param roomid:
    :param keyword:
    :return:
    """
    l1 = ['电视', '空调', '冰箱', '对讲系统', '洗衣机', '饮水机', '24小时热水', '电脑', '暖气', '厨房',
          '淋浴', '浴缸', '拖鞋', '洗漱用品', '毛巾浴巾', '卫生间', '电梯',
          '商务服务', '可以吸烟', '网络', '有线电视']

    l2 = ['可以吸烟', '允许聚会', '可养宠物']

    l3 = ['保安', '银行', '超市']

    res = get_facilities_by_id(roomid)
    out = "您好! 下面是小葱知道的房源基本情况,请陛下过目:\n"
    for k, v in res.items():
        print(k)
        if k == keyword and keyword in l1:
            if v == '-1':
                price = '没有{}'.format(keyword)
            elif v == '0':
                price = '提供免费{}'.format(keyword)
            elif v == '1':
                price = '有{},但是需要收取额外费用'.format(keyword)
            else:
                price = '不好意思,小葱无法回答您的问题!'
            out += price
            return out
        elif k == keyword and keyword in l2:
            if v == '-1':
                price = '不{}'.format(keyword)
            elif v == '0':
                price = '{}'.format(keyword)
            elif v == '1':
                price = '{},但是需要收取额外费用'.format(keyword)
            else:
                price = '不好意思,小葱无法回答您的问题!'
            out += price
            return out
        elif k == keyword and keyword in l3:
            if v == '-1':
                price = '没有{}'.format(keyword)
            elif v == '0':
                price = '有{}'.format(keyword)
            # elif v == '1':
            #     price = '有{},但是需要收取额外费用'.format(keyword)
            else:
                price = '不好意思,小葱无法回答您的问题!'
            out += price
            return out
        else:
            raise NotImplementedError("没有实现")


# print(send2tenant_message_v1(roomid='850375697', keyword='电视'))

def sort_question():
    from pyduyp.utils.fileops import sortfilebylength
    sortfilebylength(in_path='./datasets/question_sourcedata/mesg_500000.csv',
                     out_path="./datasets/question_sourcedata/ss.csv")


def jieba_cut(inputs):
    import jieba
    jieba.load_userdict("./datasets/userdict.txt")
    res = jieba.lcut(inputs)
    return " ".join(res)


def question_cut():
    import pandas as pd
    from glob import glob
    from tqdm import tqdm

    files = glob(os.path.join("./datasets/questions/question_sorted", "*.csv"))
    for f in tqdm(files):
        data = pd.read_csv(f)
        mess = data['message'].apply(jieba_cut)
        mess.to_csv("{}_cut.csv".format(f.split(".")[0]), index=None, header=None)


def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


def vsm():
    from collections import Counter

    s1 = "大床可以住几个人呢"
    s2 = "7500咋样我家亲戚也准备去要是住的好我肯定推荐"
    s12list = list(s1)
    s22list = list(s2)

    res1 = Counter(s12list)
    res2 = Counter(s22list)
    values1 = list(res1.values())
    values2 = list(res2.values())
    out = cos(values1, values2)
    print(out)


def make_test_data():
    from pyduyp.utils.utils import replace_symbol, delete_imjo

    out = []
    with open("./datasets/similar_test", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = jieba_cut(replace_symbol(delete_imjo(line))).replace("\n", "")
            if len(res) > 4:
                out.append(res)

    df = pd.DataFrame(out)
    df.to_csv("./datasets/similar_test.csv", index=None, header=None)


class MyCorpus(object):
    def __iter__(self):
        for line in open("./datasets/similar_test.csv"):
            yield line.split()


def fun():
    Corp = MyCorpus()
    dictionary = corpora.Dictionary(Corp)

    corpus = [dictionary.doc2bow(text) for text in Corp]
    print(corpus)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    index = similarities.MatrixSimilarity(corpus_tfidf)
    print(index)
    #
    # q_file = open(querypath, 'r')
    # query = q_file.readline()
    #
    # vec_bow = dictionary.doc2bow(query.split())
    # vec_tfidf = tfidf[vec_bow]
    # sims = index[vec_tfidf]
    # similarity = list(sims)
    # sim_file = open(storepath, 'w')
    # for i in similarity:
    #     sim_file.write(str(i) + '\n')
    #     sim_file.close()


def selet_top_n(inputs, n=3):
    x1 = inputs.replace("\n", "").replace(" ", "")
    return x1[:n]


def find_top2(n=3):
    import os
    out = pd.DataFrame()
    root = "./datasets/questions/question_sorted"
    files = os.listdir(root)
    i = 0
    for file in tqdm(files):
        file_name = os.path.join(root, file)
        data = pd.read_csv(file_name)
        data2save = data['message'].apply(selet_top_n)
        data2save.to_csv("./datasets/results/{}/top_{}_{}.csv".format(n, n, i), index=None, header=None)
        i += 1


def find_n_gram(n=2):
    import os
    out = pd.DataFrame()
    root = "./datasets/questions/question_sorted"
    files = os.listdir(root)
    counter = 0
    out = []
    for file in tqdm(files):
        file_name = os.path.join(root, file)
        with open(file_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                length = len(line)
                for i in range(length):
                    if i + n < length:
                        res = line[i:i + n]
                        out.append(res)
                        counter += 1
                        if counter % 10000000 == 0:
                            # out2save = list(set(out))
                            df = pd.DataFrame(out)
                            df.to_csv("./datasets/results/n_gram_all/{}/{}_gram_{}.csv".format(n, n, counter),
                                      index=None,
                                      header=None)
                            out = []


def make_top_n(n=2):
    import os
    root = "./datasets/results/n_gram/{}".format(n)
    files = os.listdir(root)
    out = []
    for file in tqdm(files):
        file_name = os.path.join(root, file)
        with open(file_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                out.append(line.strip())

    out = list(set(out))
    df = pd.DataFrame(out)
    df.to_csv("./datasets/results/n_gram/out/out_{}.csv".format(n, n), index=None)


def concont(path="./datasets/results/n_gram_all/2"):
    files = os.listdir(path)
    fw = open(os.path.join(path, 'out.txt'), 'a+')
    for file in tqdm(files):
        fr = open(os.path.join(path, file), 'r')
        lines = fr.readlines()
        fw.writelines(lines)
        fr.close()
    fw.close()


def fun3():
    from collections import Counter
    q = []
    with open("./datasets/results/n_gram_all/2/out.txt", 'r') as fr:
        lines = fr.readlines()
        log.info("=========== {} ======".format(len(lines)))
        for line in lines:
            q.append(line)

    results = pd.DataFrame()
    results.insert(0, "word", None)
    results.insert(1, "freq", None)
    i = 0
    for x, y in Counter(q).most_common(10000):
        if len(x.strip()) > 1:
            results.loc[i, 'word'] = x.strip()
            results.loc[i, 'freq'] = y
            i += 1
            if i % 5000 == 0:
                results.to_csv("./datasets/results/n_gram_all/2/freq/q_freq_{}.csv".format(i), index=None)
                results = pd.DataFrame()


def funs():
    from pyduyp.utils.fileops import sort_file_by_dict
    sort_file_by_dict("./datasets/results/22/", "out_2.csv", "top_2sort.csv", delete=False)


def make_top_n1(n=2):
    import os
    root = "./datasets/results/22"
    files = os.listdir(root)
    out = []
    for file in tqdm(files):
        file_name = os.path.join(root, file)
        with open(file_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                out.append(line.strip())

    out = list(set(out))
    df = pd.DataFrame(out)
    df.to_csv("./datasets/results/22/out_{}.csv".format(n, n), index=None)


def find_ex(inputs):
    from pyduyp.utils.utils import is_chinese
    if isinstance(inputs, str):

        if len(inputs) > 5:
            s1, sn = inputs[0], inputs[-1]
            if s1.isdigit() and sn.isdigit():
                return inputs
            else:
                d = 0
                c = 0
                s1_start = inputs[:5]
                for x in s1_start:
                    if x.isdigit():
                        d += 1
                    elif is_chinese(x):
                        c += 1
                if c > 0 and d > 0:
                    pre = d / c
                    if pre > 0.3:
                        return inputs
                else:
                    return " "
        else:
            return " "
    else:
        return " "


def find_ex_v1(inputs):
    from pyduyp.utils.utils import is_chinese
    if isinstance(inputs, str):
        if len(inputs) > 2:
            s1, sn = inputs[0], inputs[-1]
            if s1.isdigit() and sn.isdigit():
                return inputs
            else:
                return ""
        else:
            return ""
    else:
        return ""


def test_fin():
    path = './datasets/question_sourcedata'
    out = []
    files = glob(os.path.join(path, "*.csv"))
    for file in tqdm(files):
        with open(file, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                res = find_ex_v1(line)
                if isinstance(res, str):
                    if len(res) > 0 and res not in out:
                        out.append(res)


def make_answer():
    fw = open("./datasets/answer/answers.txt", "w")
    a_cache = []

    df1 = pd.read_csv("./datasets/facilities/prop.txt", usecols=[1])
    df2 = pd.read_csv("./datasets/facilities/prop_more.txt", usecols=[1])

    df12list = df1.values.tolist()
    for x in df12list:
        x = x[0]
        if x not in a_cache:
            a_cache.append(x)
    print(len(a_cache))
    df22list = df2.values.tolist()
    for x in df22list:
        x = x[0]
        if x not in a_cache:
            a_cache.append(x)

    print(len(a_cache))
    for x in tqdm(a_cache):
        if isinstance(x, str):
            fw.writelines(x + "\n")


def anlys_answer():
    fr = open("./datasets/answer/answers.txt", "r")
    lines = fr.readlines()
    out = []
    for line in lines:
        line = re.sub('[0-9]', "", line)
        res = jieba.lcut(line)
        for x in res:
            if len(x) > 1 and x not in out:
                out.append(x)

    df = pd.DataFrame(out)
    df.columns = ['word']
    df.to_csv("./datasets/answer/answers_cut.csv", index=None)


def get_all_message_by_from():
    from pyduyp.datasources.immsg import get_tenant_message_by_tenant
    save_path = 'antbot/datasets/question_45'
    df2 = pd.read_csv("./datasets/tenant_id_45.csv", usecols=['tenantid'])

    df12list = df2.values.tolist()
    i = 0
    for x in df12list:
        x = x[0]
        ret = get_tenant_message_by_tenant(x)
        log.info(ret)

        results = []
        if ret is None or len(ret) == 0:
            continue
        for row in ret:
            row = dict(row.items())
            msg = row['msg']
            if len(msg) == 0:
                continue
            results.append(msg)
            i += 1
            if i % 10000 == 0:
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(save_path, "mesg_{}.csv".format(i)), index=None, header=None)
                log.info(" =====================Success =====================")
                results = []


def seg_biaozhu():
    from jieba import posseg
    out = []
    flagss = "n, nz, v, nr"
    f = flagss.split(",")
    with open("./datasets/answer/answers_cut.csv", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = posseg.cut(line)
            for x in res:
                if x.flag in f and len(x.word) > 1:
                    out.append(x.word)

    df = pd.DataFrame(out)
    df.to_csv("./datasets/results/answer.csv", index=None)

    out = []
    data = "./datasets/key_words"
    with open(data, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if line not in out:
                out.append(line.strip())
    df = pd.DataFrame(out)
    df.to_csv("./datasets/key_words.csv", index=None)


def is_in_answer(keyword):
    with open("./datasets/answer/answers.txt", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if keyword in line:
                return True
    return False


def funx():
    from pyduyp.utils.string import is_question
    res, w = is_question("我查不到公交车站")
    print(res, w)


def xfun():
    file_name = './datasets/question_45/keywords/keywords_must_not.csv'
    data = pd.read_csv(file_name, usecols=['must', 'must_not', 'match'])
    length = len(data)
    for i in range(length):
        must, must_not, match = data.loc[i, 'must'], data.loc[i, 'must_not'], data.loc[i, 'match']
        print(must, must_not, match)


def reroute_shard(index, shard, node):
    import requests
    host = "http://localhost:9200/_cluster/allocation/explain"
    s = requests.Session()
    data = {
        "commands": [
            {
                "allocate_stale_primary": {
                    "index": index, "shard": shard, "node": node, "accept_data_loss": True
                }
            }
        ]
    }
    url = "http://localhost:9200/_cluster/reroute"
    res = s.post(url, json=data)


def get_node(line):
    import requests
    host = "http://localhost:9200/_cluster/allocation/explain"
    s = requests.Session()
    if "UNASSIGNED" in line:
        line = line.split()
        index = line[0]
        shard = line[1]
        if line[2] != "p":
            return
        body = {
            "index": index,
            "shard": shard,
            "primary": True
        }
        res = s.get(host, json=body)
        for store in res.json().get("node_allocation_decisions"):
            if store.get("store").get("allocation_id"):
                node_name = store.get("node_name")
        reroute_shard(index, shard, node_name)
    else:
        return

    with open("shards", 'rb') as f:
        map(get_node, f)


def diite():
    out = []
    with open("./datasets/cd_titie", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split("、")
            if len(line_sp) > 0:
                for x in line_sp:
                    out.append(x.strip())
    res = []
    for x in out:
        if not re.search("[0-9]", x):
            if len(x) > 0:
                res.append(x)

    for x in res:
        print(x)


def gongjiao_spyder():
    import requests
    import random
    from urllib.request import urlopen
    from urllib.request import urlretrieve
    heafers_list = []
    for i in range(10, 50):
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/{}.0'.format(i)}
        heafers_list.append(headers)

    headers = random.choice(heafers_list)

    for i in range(1, 24):
        target_url = "http://chengdu.8684.cn/sitemap{}".format(i)

        res = urlretrieve(target_url)
        print(res)


def hanLP_for_enti_v1():
    # 命名实体识别与词性标注

    from jpype import startJVM, getDefaultJVMPath, JClass

    startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/duyp/HanLp/hanlp-1.5.2.jar:/home/duyp/HanLp", "-Xms1g",
             "-Xmx1g")

    NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')

    testCases = ["天安门都江堰商品和服务迎宾宾馆北京大学",
                 "结婚的和尚未结婚的确实在干扰分词啊",
                 "买水果然后来世博园最后去世博会",
                 "中国的首都是北京",
                 "欢迎新老师生前来就餐",
                 "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
                 "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。",
                 '中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程'
                 ]

    enties = {"ntu": "大学", "nt": "机构团体名", "ntc": "公司名", "ntcb": "银行",
              "ntcf": "工厂", "ntch": "酒店宾馆", "nth": "医院", "nto": "政府机构",
              "nts": "中小学"}

    line = 1
    res = []
    for sentence in testCases:
        rows = []
        x1 = NLPTokenizer.segment(sentence)
        print(x1)


def save_question_45_to_es(index_name="question_cd"):
    from pyduyp.utils.string import is_question

    try:
        es.indices.delete(index_name)
        log.info("{} have delete ".format(index_name))
    except:
        pass
    setting = {"number_of_shards": 4, "number_of_replicas": 0}
    mapping = {"timestamp": {"enabled": "true"},
               "properties": {"logdate": {"type": "date", "format": "dd/MM/yyy HH:mm:ss"}}}

    settings = {"settings": setting, "mapping": mapping}

    es.indices.create(index=index_name, ignore=400, body=settings)
    file_dir = "./datasets/question_45/city_questions_740432.csv"

    if not os.path.exists(root):
        raise Exception("需要先整理数据")

    line_number = 0
    length = 0
    with open(file_dir, 'r') as fr:
        lines = fr.readlines()
        length += len(lines)
        all_data = []
        for line in lines:
            line_number += 1
            if is_question(line):
                all_data.append({
                    '_index': '{}'.format(index_name),
                    '_type': 'post',
                    '_id': '{}'.format(line_number),
                    '_source': {
                        'title': line
                    }
                })
            if line_number % 10000 == 0 or line_number == len(lines):
                success, _ = bulk(es, all_data, index=index_name, raise_on_error=True)
                all_data = []
                log.info("==================== success :{}/{} ====================".format(line_number, length))
    log.info("Total line number : {}".format(line_number))


def get_message_by_dates():
    results = []
    date_list = pd.date_range('2016-04-01', '2016-04-03')
    for d in date_list:
        d2time = d.to_pydatetime().strftime('%Y-%m-%d')
        messages = get_message_by_date(d2time)
        for m in messages:
            rows = {}
            if len(m) == 7 and len(m[5]) > 4:
                rows['msg'] = m[5]
                results.append(rows)

    with open("./datasets/test_message.csv", 'a+') as fa:
        fa.writelines("\n".join(results))


def get_question_by_symbol():
    res = []
    with open("./datasets/test_messages.csv", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if not re.search("我们一行", line):
                question = re.search("[\?\？]", line)
                if question:
                    res.append(line.strip().replace("\n", ""))
    df = pd.DataFrame(res)

    df.to_csv("./datasets/question.csv", index=None, header=None)


def find_question_by_name(inputs, name='几张床'):
    if isinstance(inputs, str):
        inputs2pro = inputs.replace("-", "").lower()
        p = re.compile(name, flags=re.IGNORECASE).findall(inputs2pro)
        if p:
            return inputs
        else:
            return ""
    else:
        inputs = str(inputs)
        find_question_by_name(inputs)


def select_question_by_name(name):
    results = []

    with open("./datasets/question.csv", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            for n in name:
                res = find_question_by_name(line, n)
                if len(res) > 0:
                    results.append(res.strip().replace("\n", ""))

    df = pd.DataFrame(results)
    if not os.path.exists("./questions"):
        os.makedirs("./questions")
    save_name = str(name).replace("'", "").replace("[", "").replace("]", "")
    save_name = P.get_pinyin(save_name, splitter="_")
    df.columns = ['{}'.format(save_name)]
    df.to_json("./questions/{}.json".format(save_name))


def find_all_question_by_name():
    name_list = [['几张床'], ["有浴缸"], ["能停车"], ["有房吗"], ["洗澡", "有热水"], ["做饭"], ["收费"], ["冰箱"], ["电梯"], ["wifi"]]
    for name in name_list:
        select_question_by_name(name=name)


def compute_similar():
    from gensim import corpora, models, similarities
    import jieba
    str1 = "对了，还有几个问题。自己做饭有调料吗？餐具要不要自己准备？被褥用不用自己准备？"
    str2 = "房东您好，我想问一下，您这边能做饭吗？"
    str3 = "还有就是厨房锅是不是只有一个？做饭会很慢？有没有实拍图片？"
    str4 = "除了房费，还有别的费用吗？房型是怎样的？几张床？周边有市场吗？可以做饭吧"
    str5 = "你总共有几间卧室？每间卧室分别有几张床？多大？----这个先告诉下我，我看下是不是住的下？"

    texts = []
    for s in [str1, str2, str3, str4, str5]:
        res = jieba.lcut(s)
        res = [val for val in res if len(val) > 1]
        texts.append(res)

    dictionary = corpora.Dictionary(texts)  # 'load_from_text', 'merge_with'

    dictionary.save_as_text("./datasets/dictionary.txt", sort_by_word=True)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(corpus)
    print("==========================================================")
    # （9，2 ）这个元素代表第二篇文档中id为9的单词“silver”出现了2次
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)
    print("==========================================================")
    print(tfidf.dfs)
    print(tfidf.idfs)
    print("==========================================================")

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    print(lsi.print_topics(2))
    """
    lsi的物理意义不太好解释，不过最核心的意义是将训练文档向量组成的矩阵SVD分解，
    并做了一个秩为2的近似SVD分解，可以参考那篇英文tutorail。有了这个lsi模型，
    我们就可以将文档映射到一个二维的topic空间中
    """
    corpus_lsi = lsi[corpus_tfidf]
    for doc in corpus_lsi:
        print("Doc : {}".format(doc))

    """
     好了，我们回到LSI模型，有了LSI模型，我们如何来计算文档直接的相思度，
     或者换个角度，给定一个查询Query，如何找到最相关的文档？当然首先是建索引了
     """
    index = similarities.MatrixSimilarity(lsi[corpus])
    query = "gold silver truck"
    query_bow = dictionary.doc2bow(query.lower().split())
    # 再用之前训练好的LSI模型将其映射到二维的topic空间：
    query_lsi = lsi[query_bow]
    print(query_lsi)
    # 最后就是计算其和index中doc的余弦相似度了：

    sims = index[query_lsi]
    print(list(enumerate(sims)))
    # 当然，我们也可以按相似度进行排序：
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sort_sims)
    print("===================== LDA =======================")
    """
    lda模型中的每个主题单词都有概率意义，其加和为1，值越大权重越大，物理意义比较明确，
    不过反过来再看这三篇文档训练的2个主题的LDA模型太平均了，没有说服力。
    """
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    print(lda.print_topics(2))


def tf_idf_with_scikit():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    import pickle
    from sklearn.externals import joblib
    corpus = ["I come to China to travel",
              "This is a car polupar in China",
              "I love tea and Apple ",
              "The work is to write some papers in science"]
    labels = [1, 1, 0, 0]
    tfidf2 = TfidfVectorizer()
    res = tfidf2.fit_transform(corpus)
    res2arr = res.toarray()
    df = pd.DataFrame(res2arr, dtype=np.float32)
    df['label'] = labels
    df['message'] = corpus
    df.to_csv("./datasets/df.csv", index=None)


def scikir_learn_example():
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test',
                                     categories=categories, shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.toarray().shape)
    print(type(twenty_train.target))

    clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
    print(count_vect.transform(twenty_test.data).toarray().shape)

    # test
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    predicted = clf.predict(tf_transformer.transform(count_vect.transform(twenty_test.data)))
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

    print()
    print(type(twenty_test.target_names))


def question():
    data = pd.read_table("./datasets/question_list.txt", sep=',', usecols=[1])
    data.columns = ['question']
    data.to_json("./datasets/question_list.json")


def get_file_name():
    root = "./datasets/results/search_es_results/"

    for file in os.listdir(root):
        print(file.split("_")[1].split(".")[0].strip())

