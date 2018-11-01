import urllib.request
from urllib import error
import urllib.parse
import logging
import json
import urllib


def urlhelper(url):
    user_agents = [
        'Opera/9.25 (Windows NT 5.1; U; en)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
        'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
        'Mozilla/5.0 (X11; U; linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
        'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9'
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36']

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       user_agents[6])
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        req.add_header("Cookie", "rn8tca8a5oqe72tswy7n3g08puq06vmv")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


urls = 'https://ihotel.meituan.com/hbsearch/HotelSearch?utm_medium=pc&version_name=999.9&cateId=20&attr_28=129&uuid=DE3890DBC5B402C398998C4F5BA704EA82254D14FFD89694CB3F5E204D6BC990%401539180249557&cityId=59&offset=20&limit=20&startDay=20181010&endDay=20181010&q=&sort=defaults&X-FOR-WITH=wVIhf%2BCSqUoqXZXj1vjj8vyuJtp2%2BoMzIxjuJeJQNspE4ao4wf1sy8BmP2QW5CXMNCMsKEM7Swijg%2FiFyDBR006FfFyVBQ87HjdEq8HOIlHapP1azPP5YSTFrmfM25TPiPIJBbGlUZZ2ixIvhj%2B8zQ%3D%3D'
res = urlhelper(urls)
res2dict = json.loads(res)['data']['searchresult']
print('addr' in list(json.loads(res)['data'].keys()))
for x in res2dict:
    for k, v in x.items():
        if k == 'addr':
            print(v)

