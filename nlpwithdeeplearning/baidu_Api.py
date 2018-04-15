# -*- coding: utf-8 -*-

"""
自然语言处理
"""

import hmac
import json
import hashlib
import datetime
import time
import sys
import requests
from urllib.parse import urlencode
from urllib.parse import quote
from urllib.parse import urlparse

AK = 'bAjbXhq3yfFVXeZs1oEmDNuA'
SK = 'uiGnue1gwE038nGK2rhTIyHpCN4aT30b'
appid = '11103260'


class AipBase(object):
    """
        AipBase
    """
    __accessTokenUrl = 'https://aip.baidubce.com/oauth/2.0/token'
    __reportUrl = 'https://aip.baidubce.com/rpc/2.0/feedback/v1/report'
    __scope = 'brain_all_scope'

    def __init__(self, appId=appid, apiKey=AK, secretKey=SK):
        """
            AipBase(appId, apiKey, secretKey)
        """
        self._appId = appId.strip()
        self._apiKey = apiKey.strip()
        self._secretKey = secretKey.strip()
        self._authObj = {}
        self._isCloudUser = None
        self.__client = requests
        self.__connectTimeout = 60.0
        self.__socketTimeout = 60.0
        self._proxies = {}
        self.__version = '2_2_0'

    def getVersion(self):
        """
            version
        """
        return self.__version

    def setConnectionTimeoutInMillis(self, ms):
        """
            setConnectionTimeoutInMillis
        """
        self.__connectTimeout = ms / 1000.0

    def setSocketTimeoutInMillis(self, ms):
        """
            setSocketTimeoutInMillis
        """
        self.__socketTimeout = ms / 1000.0

    def setProxies(self, proxies):
        """
            proxies
        """
        self._proxies = proxies

    def _request(self, url, data, headers=None):
        """
            self._request('', {})
        """
        try:
            result = self._validate(url, data)
            if not result:
                return result

            authObj = self._auth()
            params = self._getParams(authObj)

            data = self._proccessRequest(url, params, data, headers)
            print("requests data:{}".format(data))
            headers = self._getAuthHeaders('POST', url, params, headers)
            response = self.__client.post(url, data=data, params=params,
                                          headers=headers, verify=False, timeout=(
                    self.__connectTimeout,
                    self.__socketTimeout,
                ), proxies=self._proxies
                                          )
            obj = self._proccessResult(response.content)
            print("obj:{}".format(obj))
            if not self._isCloudUser and obj.get('error_code', '') == 110:
                authObj = self._auth(True)
                params = self._getParams(authObj)
                response = self.__client.post(url, data=data, params=params,
                                              headers=headers, verify=False, timeout=(
                        self.__connectTimeout,
                        self.__socketTimeout,
                    ), proxies=self._proxies
                                              )
                obj = self._proccessResult(response.content)
                print("103 obj:{}".format(obj))
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            return {
                'error_code': 'SDK108',
                'error_msg': 'connection or read data timeout',
            }

        return obj

    def _validate(self, url, data):
        """
            validate
        """
        return True

    def _proccessRequest(self, url, params, data, headers):
        """
            参数处理
        """
        params['aipSdk'] = 'python'
        params['aipVersion'] = self.__version
        return data

    def _proccessResult(self, content):
        """
            formate result
        """
        return json.loads(content.decode()) or {}

    def _auth(self, refresh=False):
        """
            api access auth
        """

        # 未过期
        if not refresh:
            tm = self._authObj.get('time', 0) + int(self._authObj.get('expires_in', 0)) - 30
            if tm > int(time.time()):
                return self._authObj

        obj = self.__client.get(self.__accessTokenUrl, verify=False, params={
            'grant_type': 'client_credentials',
            'client_id': self._apiKey,
            'client_secret': self._secretKey,
        }, timeout=(self.__connectTimeout, self.__socketTimeout), proxies=self._proxies).json()

        self._isCloudUser = not self._isPermission(obj)
        obj['time'] = int(time.time())
        self._authObj = obj
        return obj

    def _isPermission(self, authObj):
        """
            check whether permission
        """
        scopes = authObj.get('scope', '')
        return self.__scope in scopes.split(' ')

    def _getParams(self, authObj):
        """
            api request http url params
        """
        params = {}
        if not self._isCloudUser:
            params['access_token'] = authObj['access_token']
        print("params:{}".format(params))
        return params

    def _getAuthHeaders(self, method, url, params=None, headers=None):
        """
            api request http headers
        """
        headers = headers or {}
        params = params or {}
        if not self._isCloudUser:
            return headers
        urlResult = urlparse(url)
        for kv in urlResult.query.strip().split('&'):
            if kv:
                k, v = kv.split('=')
                params[k] = v

        # UTC timestamp
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        headers['Host'] = urlResult.hostname
        headers['x-bce-date'] = timestamp
        version, expire = '1', '1800'

        # 1 Generate SigningKey
        val = "bce-auth-v%s/%s/%s/%s" % (version, self._apiKey, timestamp, expire)
        signingKey = hmac.new(self._secretKey.encode('utf-8'), val.encode('utf-8'),
                              hashlib.sha256
                              ).hexdigest()

        # 2 Generate CanonicalRequest
        # 2.1 Genrate CanonicalURI
        canonicalUri = quote(urlResult.path)
        # 2.2 Generate CanonicalURI: not used here
        # 2.3 Generate CanonicalHeaders: only include host here

        canonicalHeaders = []
        for header, val in headers.items():
            canonicalHeaders.append(
                '%s:%s' % (
                    quote(header.strip(), '').lower(),
                    quote(val.strip(), '')
                )
            )
        canonicalHeaders = '\n'.join(sorted(canonicalHeaders))

        # 2.4 Generate CanonicalRequest
        canonicalRequest = '%s\n%s\n%s\n%s' % (
            method.upper(),
            canonicalUri,
            '&'.join(sorted(urlencode(params).split('&'))),
            canonicalHeaders
        )

        # 3 Generate Final Signature
        signature = hmac.new(signingKey.encode('utf-8'), canonicalRequest.encode('utf-8'), hashlib.sha256).hexdigest()
        headers['authorization'] = 'bce-auth-v%s/%s/%s/%s/%s/%s' % (
            version, self._apiKey, timestamp, expire, ';'.join(headers.keys()).lower(), signature)
        print("header: {}".format(headers))
        return headers

    def report(self, feedback):
        """
            数据反馈
        """
        data = {'feedback': feedback}
        return self._request(self.__reportUrl, data)


class AipNlp(AipBase):
    """
    自然语言处理
    """

    __lexerUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/lexer'
    __lexerCustomUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/lexer_custom'
    __depParserUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/depparser'
    __wordEmbeddingUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/word_emb_vec'
    __dnnlmCnUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/dnnlm_cn'
    __wordSimEmbeddingUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/word_emb_sim'
    __simnetUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/simnet'
    __commentTagUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v2/comment_tag'
    __sentimentClassifyUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify'
    __keywordUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/keyword'
    __topicUrl = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/topic'

    def _proccessResult(self, content):
        """
            formate result
        """
        return json.loads(str(content, 'gbk')) or {}

    def _proccessRequest(self, url, params, data, headers):
        """
            _proccessRequest
        """
        return json.dumps(data, ensure_ascii=False).encode('gbk')

    def lexer(self, text, options=None):
        """
            词法分析
        """
        options = options or {}
        data = {'text': text}
        data.update(options)
        return self._request(self.__lexerUrl, data)

    def lexerCustom(self, text, options=None):
        """
            词法分析（定制版）
        """
        options = options or {}
        data = {'text': text}
        data.update(options)
        return self._request(self.__lexerCustomUrl, data)

    def depParser(self, text, options=None):
        """
            依存句法分析
        """
        options = options or {}
        data = {'text': text}
        data.update(options)
        return self._request(self.__depParserUrl, data)

    def wordEmbedding(self, word, options=None):
        """
            词向量表示
        """
        options = options or {}
        data = {'word': word}
        data.update(options)
        return self._request(self.__wordEmbeddingUrl, data)

    def dnnlm(self, text, options=None):
        """
            DNN语言模型
        """
        options = options or {}
        data = {'text': text}
        data.update(options)
        return self._request(self.__dnnlmCnUrl, data)

    def wordSimEmbedding(self, word_1, word_2, options=None):
        """
            词义相似度
        """
        options = options or {}
        data = {'word_1': word_1, 'word_2': word_2}
        data.update(options)
        return self._request(self.__wordSimEmbeddingUrl, data)

    def simnet(self, text_1, text_2, options=None):
        """
            短文本相似度
        """
        options = options or {}
        data = {'text_1': text_1, 'text_2': text_2}
        data.update(options)
        print(data)
        return self._request(self.__simnetUrl, data)

    def commentTag(self, text, options=None):
        """
            评论观点抽取
        """
        options = options or {}
        data = {'text': text}
        data.update(options)
        return self._request(self.__commentTagUrl, data)

    def sentimentClassify(self, text, options=None):
        """
            情感倾向分析
        """
        options = options or {}
        data = {'text': text}
        data.update(options)
        return self._request(self.__sentimentClassifyUrl, data)

    def keyword(self, title, content, options=None):
        """
            文章标签
        """
        options = options or {}
        data = {'title': title, 'content': content}
        data.update(options)
        return self._request(self.__keywordUrl, data)

    def topic(self, title, content, options=None):
        """
            文章分类
        """
        options = options or {}
        data = {'title': title, 'content': content}
        data.update(options)
        return self._request(self.__topicUrl, data)


if __name__ == '__main__':
    baiduapi = AipNlp()
    res = baiduapi.wordEmbedding(word='你好')
    res1 = baiduapi.simnet(text_1='今天又房吗', text_2='请问今天还有房间吗？')['score']
