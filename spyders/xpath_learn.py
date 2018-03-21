from lxml import etree
import urllib.request
import re
import os
from tqdm import trange, tqdm
import json
from urllib.request import urlopen, quote
import requests, csv
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error  # 回归模型使用
from io import BytesIO
import gzip
import demjson
from urllib import error
import logging
import tensorflow as tf
from tensorflow.contrib import rnn

from antcolony.logger.log import log


def randomforest(method='regression', data_name='datali.csv'):
    data = pd.read_csv(data_name)
    train_target = data['Problem types']
    del data["Problem types"]
    names = data.columns
    train_data = data.values
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2,
                                                        random_state=0)
    print("训练集X:{},训练集Y：{}, 测试集X：{},测试集Y：{}".format(len(X_train), len(y_train), len(X_test), len(y_test)))
    res = []
    score_list = []
    mse_list = []
    acc_list = []
    for tree in [50, 100, 200, 300, 400, 500]:
        if method == 'regression':
            rf = RandomForestRegressor(n_estimators=tree,
                                       criterion="mse",
                                       max_depth=None,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.,
                                       max_features="auto",
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.,
                                       min_impurity_split=None,
                                       bootstrap=True,
                                       oob_score=False,
                                       n_jobs=1,
                                       random_state=None,
                                       verbose=0,
                                       warm_start=False)
        else:
            rf = RandomForestClassifier(n_estimators=tree,
                                        criterion="gini",
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.,
                                        max_features="auto",
                                        max_leaf_nodes=None,
                                        min_impurity_decrease=0.,
                                        min_impurity_split=None,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=1,
                                        random_state=None,
                                        verbose=0,
                                        warm_start=False,
                                        class_weight=None)
        rf.fit(X_train, y_train)
        joblib.dump(rf, "rf.model")
        y_pred = rf.predict(X_test)
        predicts = pd.DataFrame()
        predicts['true'] = y_test
        predicts['pred'] = y_pred
        predicts.to_csv("predicts_{}.csv".format(tree), index=None)
        acc_val = abs(y_test - y_pred)
        k = 0
        for x in acc_val:
            if x == 0:
                k += 1
            else:
                k += 0
        acc = k / len(y_test)
        print("acc: {}".format(acc))
        acc_rows = {'acc': acc, "ntree": tree}
        acc_list.append(acc_rows)
        if method == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mse_rows = {'mse': mse, "ntree": tree}
            mse_list.append(mse_rows)
        ma = confusion_matrix(y_test, y_pred)
        print("混淆矩阵：\n {}".format(ma))
        score = rf.score(X_test, y_test)
        score_rows = {'score': score, "ntree": tree}
        score_list.append(score_rows)
        results = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                         reverse=True)
        for score in results:
            rows = {"name": score[1], "score": score[0]}
            res.append(rows)
        df = pd.DataFrame(res)
        df.to_csv("results_{}.csv".format(tree), index=None)
    if method == 'regression':
        dfmse = pd.DataFrame(mse_list)
        dfmse.to_csv("mse.csv", index=None)
    # score
    dftree = pd.DataFrame(score_list)
    dftree.to_csv("score.csv", index=None)
    # acc
    dfacc = pd.DataFrame(acc_list)
    dfacc.to_csv("acc.csv", index=None)


def timestamp2localtime(timestamp, method='first'):
    if method == 'first':
        tllocal = time.localtime(timestamp)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", tllocal)
        return dt
    if method == 'second':
        tllocal = time.localtime(timestamp)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", tllocal)
        dtime = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        return dtime
    if method == 'third':
        tllocal = datetime.fromtimestamp(timestamp)
        return tllocal
    if method == 'four':
        tllocal = time.localtime(timestamp)
        year, mon, mday, hour, minits, second = tllocal[0], tllocal[1], tllocal[2], tllocal[3], tllocal[4], tllocal[5],
        return year, mon, mday, hour, minits, second


def geturlcontent(url):
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
    req.add_header("Accept", "*/*")
    req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
    try:
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html
    except Exception as e:
        log.warning("{}".format(e))


def xiaozhu_spydr():
    citys = ['gz', 'sanya', 'qd', 'lijiang']
    for city in citys:
        n = 0
        for page in range(30):
            url = 'http://{}.xiaozhu.com/search-duanzufang-p{}-0/'.format(city, page)
            fw = open("./localdatasets/{}.txt".format(city), 'a+', encoding="utf-8")
            html = geturlcontent(url)
            selector = etree.HTML(html, parser=None, base_url=None)
            context = selector.xpath('//a/@href')
            for x in context:
                pattern = "http://{}.xiaozhu.com/fangzi/\d+.html".format(city)
                res = re.compile(pattern).findall(x)
                if res:
                    fangzilink = res[0]
                    fw.writelines(fangzilink + "\n")
                    n += 1
        print(n)


def abuyun():
    from urllib import request

    # 要访问的目标页面
    targetUrl = "http://test.abuyun.com/proxy.php"
    # 代理服务器
    proxyHost = "http-pro.abuyun.com"
    proxyPort = "9010"
    # 代理隧道验证信息
    proxyUser = "H01234567890123P"
    proxyPass = "0123456789012345"
    proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
        "host": proxyHost,
        "port": proxyPort,
        "user": proxyUser,
        "pass": proxyPass,
    }
    proxy_handler = request.ProxyHandler({
        "http": proxyMeta,
        "https": proxyMeta,
    })
    auth = request.HTTPBasicAuthHandler()
    opener = request.build_opener(proxy_handler, auth, request.HTTPHandler)
    opener = request.build_opener(proxy_handler)
    opener.addheaders = [("Proxy-Switch-Ip", "yes")]
    request.install_opener(opener)
    resp = request.urlopen(targetUrl).read()
    print(resp)


def contact():
    fw = open("./localdatasets/xiaozhu_fangyuan.txt", 'a+')

    for file in os.listdir("./localdatasets/"):
        file_name = os.path.join("./localdatasets", file)
        with open(file_name, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                fw.writelines(line.strip() + "\n")


def get_zhenguo_url():
    root = "https://phoenix.meituan.com/"
    citys = ['shanghai', 'beijing', 'chengdu', 'xian']
    for city in citys:
        fw = open("./localdatasets/zhenguo_{}.txt".format(city), 'a+')
        urls = "https://phoenix.meituan.com/{}".format(city)
        for i in trange(60):
            url = urls + "/pn{}".format(i)
            html = geturlcontent(url)
            if not html:
                continue
            selector = etree.HTML(html, parser=None, base_url=None)
            context = selector.xpath('//a/@href')
            for x in context:
                pattern = "housing/\d+".format(city)
                res = re.compile(pattern).findall(x)
                if res:
                    fangzilink = root + res[0]
                    fw.writelines(fangzilink + "\n")


def zhenguo_spyder():
    res = []
    citys = ['shanghai', 'chengdu', 'xian']
    for city in citys:
        with open("./localdatasets/zhenguo_{}.txt".format(city), 'r') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                html = geturlcontent(line)
                if not html:
                    continue
                soup = BeautifulSoup(html, 'lxml')
                try:
                    resp = soup.findAll('p', attrs={"class": "product-detail__location"})[0].text
                except:
                    continue
                try:
                    name = soup.findAll('div', attrs={"class": "detail__breadcrumb"})[0].text
                except:
                    continue
                try:
                    price = soup.findAll('header', attrs={'class': "product-price"})[0].text
                except:
                    continue

                rows = {"address": resp.replace("\n", ""), "name": name.replace("\n", ""), "price": price}
                print(rows)
                res.append(rows)
            df = pd.DataFrame(res)
            df.to_csv("./localdatasets/zhenguo_{}_results.csv".format(city), index=None)


def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = '你申请的密钥***'
    add = quote(address)  # 由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode()  # 将其他编码的字符串解码成unicode
    temp = json.loads(res)  # 对json数据进行解析
    return temp


def str2timestamp(inputs, fmat='%Y-%m-%d %H:%M'):
    return time.mktime(time.strptime(inputs, fmat))


def post_method():
    import urllib.request
    rooturl = 'https://www.cailianpress.com/'
    html = geturlcontent(rooturl)
    soup = BeautifulSoup(html, "lxml")
    newsLeft = soup.findAll('div', attrs={'class': 'newsLeft'})
    newsRight = soup.findAll('div', attrs={'class': 'newsRight'})
    dates = soup.findAll("div", attrs={"class": 'time'})[0].text
    print(dates)
    for i in range(len(newsLeft)):
        content = newsRight[i].findAll('p', attrs={'class': 'contentC'})[0].text
        href = re.compile('href="/roll/\d+"').search(str(newsRight[i])).group(0)
        ctime = newsLeft[i].findAll('div', attrs={'class': 'cTime'})[0].text
        lasttime = newsLeft[len(newsLeft) - 1].findAll("div", attrs={'class': "cTime"})[0].text
        fmat = time.mktime(time.strptime('{} {}'.format(dates, ctime), '%Y-%m-%d %H:%M'))

        cookie = "3AB9D23F7A4B3C9B=RFOP5Z6MSHJCCCTIFOZB66HNKX4LS6U6OIBF4JL7QBT2UCYUYTABR6GBIUB3DNNECZF3UTMPKIRSP5ASEEJKNUSONI; __jdu=1505113579138122636331; __jda=122270672.1505113579138122636331.1505113579.1505119561.1520504433.3; __jdc=122270672; __jdv=122270672|direct|-|none|-|1520504433202; PCSYCityID=1; ipLoc-djd=1-72-4137-0; areaId=1; __jdb=122270672.3.1505113579138122636331|3.1520504433“"
        sign_get = ''
        host = 'www.cailianpress.com'
        referer = "https://www.cailianpress.com/"
        headers = {
            'User-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
            'Cookie': cookie,
            'Connection': 'keep-alive',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, sdch',
            'Accept-Language': 'zh-CN,zh;q=0.8',
            'Host': host,
            'Referer': referer,
        }
        url = "https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv538&productId=4687958&score=0&sortType=5&page=5&pageSize=10&isShadowSku=0&rid=0&fold=1"

        include_urls = ['http://www.jianshu.com/notes/{}/included_collections?page={}'.format(id, str(i)) for i in
                        range(1, 4)]
        for include_url in include_urls:
            req = urllib.request.Request(include_url, None, headers)
            response = urllib.request.urlopen(req)
            print(response.status)
            response_code = response.status
            if response_code != 200:
                continue
            the_page = response.read()
            buff = BytesIO(the_page)  # 把content转为文件对象
            f = gzip.GzipFile(fileobj=buff).read().decode('utf-8')
            print(type(f))
            # json_dict = json.loads(f)
            json_dict = demjson.decode(f, encoding='utf-8')
            print(type(json_dict), json_dict.keys())
            for item in json_dict['data']['roll_data']:
                rows = {'title': item['title'], 'content': item['content'], 'modified_time': item['modified_time'],
                        'ctime': item['ctime']}
                print(rows)


def jd_spyder():
    import urllib.request

    cookie = "3AB9D23F7A4B3C9B=RFOP5Z6MSHJCCCTIFOZB66HNKX4LS6U6OIBF4JL7QBT2UCYUYTABR6GBIUB3DNNECZF3UTMPKIRSP5ASEEJKNUSONI; __jdu=1505113579138122636331; __jda=122270672.1505113579138122636331.1505113579.1505119561.1520504433.3; __jdc=122270672; __jdv=122270672|direct|-|none|-|1520504433202; PCSYCityID=1; ipLoc-djd=1-72-4137-0; areaId=1; __jdb=122270672.3.1505113579138122636331|3.1520504433“"
    host = 'www.cailianpress.com'
    referer = "https://www.cailianpress.com/"
    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
        'Cookie': cookie,
        'Connection': 'keep-alive',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Host': host,
        'Referer': referer,
    }
    include_urls = ["https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv538&productId=4687958&score=0&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1".format(page) for page in range(1, 100)]

    for include_url in include_urls:
        req = urllib.request.Request(include_url, None, headers)
        response = urllib.request.urlopen(req)
        response_code = response.status
        if response_code != 200:
            continue
        the_page = response.read().decode('latin-1')
        print(the_page.keys())
        exit(1)


jd_spyder()

