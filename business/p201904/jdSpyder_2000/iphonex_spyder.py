import pandas as pd
import urllib.request as req
import json
import sys
import time
import random
import os

print(sys.getdefaultencoding())


class JDCommentsCrawler:

    def __init__(self, productId=None, callback=None, start=1, page=9999, score=0, sortType=5, pageSize=10):
        self.productId = productId  # 商品ID
        self.score = score  # 评论类型（好：3、中：2、差：1、所有：0）
        self.sortType = sortType  # 排序类型（推荐：5、时间：6）
        self.pageSize = pageSize  # 每页显示多少条记录（默认10）
        self.callback = callback  # 回调函数，每个商品都不一样
        self.page = page
        self.start = start
        self.locationLink = 'https://sclub.jd.com/comment/productPageComments.action'
        self.paramValue = {
            'callback': self.callback,
            'productId': self.productId,
            'score': self.score,
            'sortType': self.sortType,
            'pageSize': self.pageSize,
        }
        self.locationUrl = None

    def paramDict2Str(self, params):
        str1 = ''
        for p, v in params.items():
            str1 = str1 + p + '=' + str(v) + '&'
        return str1

    def concatLinkParam(self):
        self.locationUrl = self.locationLink + '?' + self.paramDict2Str(self.paramValue) + 'isShadowSku=0&fold=1&page=0'
        # print(self.locationUrl)

    def requestMethod(self):
        headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Referer': 'https://item.jd.com/%d.html' % (self.productId),
            'Host': 'sclub.jd.com'
        }
        reqs = req.Request(self.locationUrl, headers=headers)
        return reqs

    def showList(self):
        request_m = self.requestMethod()
        conn = req.urlopen(request_m)
        return_str = conn.read().decode('gbk')
        return_str = return_str[len(self.callback) + 1:-2]
        return json.loads(return_str)

    def requestMethodPage(self, p):
        # 伪装浏览器 ，打开网站
        headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Referer': 'https://item.jd.com/%d.html' % self.productId,
            'Host': 'sclub.jd.com'
        }
        url = self.locationUrl[:-1] + str(p)
        reqs = req.Request(url, headers=headers)
        return reqs

    def showListPage(self, p):
        request_m = self.requestMethodPage(p)
        conn = req.urlopen(request_m)
        return_str = conn.read().decode('gbk')
        return_str = return_str[len(self.callback) + 1:-2]
        return json.loads(return_str)

    def save_csv(self, df, p):
        # 保存文件
        if not os.path.exists("./data"):
            os.makedirs('./data')
        df.to_csv(path_or_buf='./data/jd_%d.csv' % p, encoding='gbk', index=None)

    def crawler(self):
        # 把抓取的数据存入CSV文件，设置时间间隔，以免被屏蔽
        dfs = []
        for p in range(self.start, self.page):
            try:
                json_info = self.showListPage(p)
                tmp_list = []
                productCommentSummary = json_info['productCommentSummary']
                productId = productCommentSummary['productId']
                comments = json_info['comments']
                for com in comments:
                    tmp_list.append(
                        [com['id'], productId, com['guid'], com['content'], com['creationTime'], com['referenceId'],
                         com['referenceTime'], com['score'], com['nickname'], com['userLevelName'], com['isMobile'],
                         com['userClientShow']])
                df = pd.DataFrame(tmp_list,
                                  columns=['comment_id', 'product_id', 'guid', 'content', 'create_time', 'reference_id',
                                           'reference_time', 'score', 'nickname', 'user_level', 'is_mobile', 'user_client'])
                self.save_csv(df, p)
            except:
                continue
            dfs.append(df)
            time.sleep(random.randint(31, 52))
        final_df = pd.concat(dfs, ignore_index=True)
        self.save_csv(final_df, self.page)
        print(final_df.shape)


def jdComment():
    # 设置关键变量
    page = 100  # 页数
    productId = 5089253  # 商品ID, iphoneX
    callback = 'fetchJSON_comment98vv782'  # 回调函数
    JDC = JDCommentsCrawler(productId, callback, start=page)
    JDC.concatLinkParam()
    JDC.crawler()


if __name__ == '__main__':
    jdComment()


