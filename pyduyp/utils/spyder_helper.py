import urllib.request
import requests
import urllib.request
from urllib import error
import logging


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


def urlhelper(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 6.1; WOW64)"
                       " AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/45.0.2454.101 Safari/537.36")
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')

        return html
    except error.URLError as e:
        logging.warning("{}".format(e))

