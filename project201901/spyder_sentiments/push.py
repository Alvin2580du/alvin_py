import re
import requests
from pyquery import PyQuery as pq
import json
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
import urllib.request
from urllib import error
import logging
import time

# =============================大众点评===============================

header_pinlun = {
    # 'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Connection': 'keep-alive',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
    'Cookie': '_lxsdk_cuid=16898f67cb8c8-0a6fb581dd372b-b781636-1fa400-16898f67cb9c8; _lxsdk=16898f67cb8c8-0a6fb581dd372b-b781636-1fa400-16898f67cb9c8; _hc.v=e6945489-db7d-5e88-49b0-7ec0239cb213.1548754519; aburl=1; cy=22; cye=jinan; Hm_lvt_dbeeb675516927da776beeb1d9802bd4=1548757678; QRCodeBottomSlide=hasShown; dper=66043df9ff2abe702d1b7e31f1b7611e81800c7d75b3f34e6d36004b8961156441a52048f4bd19f81e9ab6312011e476e79b1710cd45b9b3694c353855642f7e7b21d633756c63eb5f9f56c78940b90176a3b47867cce24782889fb09d39ba98; ll=7fd06e815b796be3df069dec7836c3df; ua=%E8%8B%8F; ctu=eb999170f60aa828224f0aeeb59541dc266df19b0fd662442a1f70dfdd50db77; _lxsdk_s=168fe71ace2-451-a74-056%7C%7C805'
}

header_css = {
    'Host': 's3plus.meituan.net',
    'Accept-Encoding': 'gzip',
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'

}


def svg_dict(csv_html):
    svg_text_r = r'<text x="0" y="(.*?)">(.*?)</text>'
    svg_text_re = re.findall(svg_text_r, csv_html)
    dict_avg = {}
    # 生成svg加密字体库字典
    for data in svg_text_re:
        flag = str(int(data[0]) - 23)
        dict_avg[flag] = list(data[1])
    return dict_avg


def svg_text(url):
    html = requests.get(url)
    dict_svg = svg_dict(html.text)
    return dict_svg


def css_get(doc):
    css_link = "http:" + doc("head > link:nth-child(11)").attr("href")
    background_link = requests.get(css_link, headers=header_css)
    r = r'background-image: url(.*?);'
    matchObj = re.compile(r, re.I)
    svg_link = matchObj.findall(background_link.text)[1].replace(")", "").replace("(", "http:")
    dict_avg_text = svg_text(svg_link)
    dict_css = css_dict(background_link.text)
    return dict_avg_text, dict_css


# 4-生成css字库字典
def css_dict(html):
    css_text_r = r'.(.*?){background:(.*?)px (.*?)px;}'
    css_text_re = re.findall(css_text_r, html)
    dict_css = {}
    for data in css_text_re:
        x = str(int(float(data[1]) / -14))
        y = re.findall(r"-(.+?).0", data[2])[0]
        dict_css[data[0]] = (x, y)
    return dict_css


# 5-最终评论汇总
def css_decode(css_dict_text, svg_dict, pinglun_html):
    pinglun_text = pq(pinglun_html.replace('<span class="', ',').replace('"/>', ",").replace('">', ",")).text()
    pinglun_list = [x for x in pinglun_text.split(",") if x != '']
    pinglun_str = []
    for msg in pinglun_list:
        # 如果有加密标签
        if msg in css_dict_text:
            x = int(css_dict_text[msg][0])
            y = str(css_dict_text[msg][1])
            try:
                pinglun_str.append(svg_dict[y][x])
            except:
                pinglun_str.append('')

        else:
            pinglun_str.append(msg.replace("\n", ""))
    str_pinglun = ""
    for x in pinglun_str:
        str_pinglun += x
    return str_pinglun


def build_dazhong(id, name):
    save = []
    for page in range(1, 40):
        try:
            url = "http://www.dianping.com/shop/{0}/review_all/p{1}".format(id, page)
            html = requests.get(url, headers=header_pinlun)
            doc = pq(html.text)
            # 解析每条评论
            pinglunLi = doc("div.reviews-items > ul > li").items()
            dict_svg_text, dict_css_x_y = css_get(doc)
            for data in pinglunLi:
                information = {}
                # 关键部分，评论HTML,待处理，评论包含隐藏部分和直接展示部分，默认从隐藏部分获取数据，没有则取默认部分。（查看更多）
                pinglun = data("div.review-words.Hide").html()
                try:
                    len(pinglun)
                except:
                    pinglun = data("div.review-words").html()
                comment = css_decode(dict_css_x_y, dict_svg_text, pinglun)
                information['pinglun'] = comment
                information['address'] = name
                save.append(information)
                print(information)
        except:
            break
    df = pd.DataFrame(save)
    df.to_csv("dazhong_Results.csv", index=None)
    print(df.shape)


# =============================airbnb===============================

def airbnb():
    list = []
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400 QQBrowser/9.7.12661.400'
    }

    for m in range(0, 36, 18):
        url = 'https://www.airbnb.cn/api/v2/explore_tabs?version=1.4.5&satori_version=1.1.3&_format=for_explore_search_web&experiences_per_grid=20&items_per_grid=18&guidebooks_per_grid=20&auto_ib=true&fetch_filters=true&has_zero_guest_treatment=true&is_guided_search=true&is_new_cards_experiment=true&luxury_pre_launch=false&query_understanding_enabled=false&show_groupings=true&supports_for_you_v3=true&timezone_offset=480&client_session_id=c1dc98aa-5de6-422b-99eb-3ed164e09dc6&metadata_only=false&is_standard_search=true&refinement_paths[]=/homes&selected_tab_id=home_tab&click_referer=t:SEE_ALL|sid:29141e94-7a48-4a7e-a794-0b53c13c272d|st:MAGAZINE_HOMES&title_type=MAGAZINE_HOMES&allow_override[]=&s_tag=PAJAldIi&section_offset=6&items_offset={0}&screen_size=large&query=%E8%8B%8F%E5%B7%9E%E5%8C%97%E7%AB%99&_intents=p1&key=d306zoyjsyarp7ifhu67rjxn52tv0t20&currency=CNY&locale=zh'.format(
            m)
        url_json = requests.get(url, headers=headers).text
        url_json = json.loads(url_json)
        url_list = url_json['explore_tabs'][0]['home_tab_metadata']['remarketing_ids']
        for url in url_list:
            list.append(url)

    print(len(list))

    save = []
    for id in list:
        url = 'https://www.airbnb.cn/rooms/{0}?location=%E8%8B%8F%E5%B7%9E%E5%8C%97%E7%AB%99&s=PAJAldIi&guests=1&adults=1'.format(
            id)
        html = requests.get(url, headers=headers).text
        key = re.findall(r"key:(.+?)}", html.replace('&quot;', ''))[0]
        id = re.findall(r"rooms/(\d+)?", url)[0]
        for number in range(0, 100, 7):
            try:
                url2 = 'https://www.airbnb.cn/api/v2/reviews?key={0}&currency=CNY&locale=zh&listing_id={1}&role=guest&_format=for_p3&_limit=7&_offset={2}&_order=language_country'.format(
                    key, id, number)
                print(url2)
                time.sleep(2)
                html2 = requests.get(url2, headers=headers).text
                html2 = json.loads(html2)
                try:
                    comment_list = html2['reviews']
                except:
                    time.sleep(2)
                    comment_list = html2['reviews']
                for m in comment_list:
                    information = {'comment': m['comments'], 'address': 'airbnb'}
                    save.append(information)
                    print(information)
            except:
                break

    df = pd.DataFrame(save)
    df.to_csv("airbnb_Results.csv", index=None)
    print(df.shape)


# =============================房天下===============================

def get_fangtianxia_urls(file_name, url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400 QQBrowser/9.7.12661.400'
    }
    text = requests.get(url, headers=headers).text
    soup = BeautifulSoup(text, 'html.parser')
    divlist = soup.find(id='newhouse_loupai_list').find_all(name='li')
    for m in divlist:
        try:
            href = m.find(class_='nlcd_name').find(name='a').attrs['href']
        except:
            continue
        print(href)
        with open("{}.txt".format(file_name), 'a+', encoding='utf-8') as f:
            f.write(href + '\n')


def build_fangtianxia(file_name):
    save = []
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.104 Safari/537.36 Core/1.53.4295.400 QQBrowser/9.7.12661.400'
    }
    with open("{}.txt".format(file_name), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            url = 'https:' + line + 'dianping'
            print(url)
            html = requests.get(url, headers=headers)
            html.encoding = "gbk"
            soup = BeautifulSoup(html.text, 'html.parser')
            divlist = soup.find_all(class_='comm_list')
            for divtag in divlist:
                comm_dict = {}
                try:
                    comm_dict['comment'] = divtag.find(class_='comm_list_con').text.replace(' ', '').replace('\n', '')
                    comm_dict['address'] = '房天下'
                    save.append(comm_dict)
                    print(comm_dict)
                except:
                    continue

            try:
                # https://tianhaojlc.fang.com/house/ajaxrequest/dianpingList_201501.php
                url2 = 'https://taibeigongguan.fang.com/house/ajaxrequest/dianpingList_201501.php'
                code = re.findall(r"dianpingNewcode = \"(\d+)", html.text)[0]
                for n in range(2, 20):
                    data = {
                        'city': '苏州',
                        'dianpingNewcode': code,
                        'ifjiajing': '0',
                        'page': n,
                        'pagesize': '20'
                    }
                    html2 = requests.post(url2, data=data, headers=headers).text
                    try:
                        html2 = json.loads(html2)
                        comentlist = html2['list']
                        for comment in comentlist:
                            comm_dict = {'comment': comment['content'], 'address': '房天下'}
                            save.append(comm_dict)
                            print(comm_dict)
                    except:
                        break
            except:
                pass

    df = pd.DataFrame(save)
    df.to_csv("{}_Results.csv".format(file_name.replace(".txt", "")), index=None)
    print(df.shape)


# =============================百度贴吧===============================

def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


ug = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36",
      'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
      ]


def urlhelper(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.81 Safari/537.36")
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html

    except error.URLError as e:
        logging.warning("{}".format(e))


def build_tieba_links():
    # 第一步先获取贴吧的全部链接，保存到文件links.csv 文件中，一共保存569个帖子
    links = []
    for page in range(0, 650, 50):
        try:
            patt = 'rel="noreferrer" href="/p/\d+'
            urls = "http://tieba.baidu.com/f?kw=%E8%8B%8F%E5%B7%9E%E9%AB%98%E9%93%81%E6%96%B0%E5%9F%8E&ie=utf-8&pn={}".format(
                page)
            html = urlhelper(urls)
            href = re.compile(patt).findall(html)
            for x in href:
                tmp = x.replace('rel="noreferrer" href="/p/', "")
                link = "https://tieba.baidu.com/p/{}".format(tmp)
                links.append(link)
        except Exception as e:
            continue

    df = pd.DataFrame(links)
    df.to_csv("tieba_links.csv", index=None)
    print(df.shape)


def is_Chinese(word):
    if '\u4e00' <= word <= '\u9fff':
        return True
    return False


def is_num(num):
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if num in nums:
        return True
    return False


def process_comments(inputs):
    biaodian = ['。', '，', '!', '！', '?', '？', '；', '；', '、']
    tmp = str(inputs).replace("\n", "")
    out = []
    for x in tmp.split(" "):
        if '<br' in x:
            for i in x:
                if is_Chinese(i) or i in biaodian or str(i).isnumeric():
                    out.append(i)
    return "".join(out)


def build_tieba(file_name="tieba_links.csv"):
    comments = []
    tieba = pd.read_csv(file_name)
    for x, y in tieba.iterrows():
        time.sleep(3)
        try:
            urls = y[0]
            htmls = urlhelper(urls)
            soup = BeautifulSoup(htmls, 'lxml')
            total = soup.findAll('div', attrs={"class": 'p_postlist'})
            for one in total:
                try:
                    ret = process_comments(one)
                    rows = {"address": '贴吧', 'comment': ret}
                    comments.append(rows)
                    print(rows)
                except:
                    continue
        except Exception as e:
            continue

    df = pd.DataFrame(comments)
    df.to_csv("comments.csv", index=None)
    print(df.shape)


if __name__ == '__main__':
    method = 'airbnb'

    if method == 'build_dazhong':
        ids = {'121819266': "高铁苏州北站南广场",
               '5260820': "苏州北站",
               '76882046': "圆融购物中心",
               '83280598': "高铁北站-综合服务楼"}

        for k, v in ids.items():
            build_dazhong(id=k, name=v)

    if method == 'airbnb':
        airbnb()

    if method == 'get_fangtianxia_urls':
        s = {"a73": "别墅", "a77": "住宅", "a75": "写字楼", "a72": "商铺"}
        for k, v in s.items():
            url = "https://suzhou.newhouse.fang.com/house/s/xiangcheng/{}/".format(k)
            get_fangtianxia_urls(file_name='{}'.format(v), url=url)

    if method == 'build_fangtianxia':
        file = {"a73": "别墅", "a77": "住宅", "a75": "写字楼", "a72": "商铺"}
        for k, v in file.items():
            build_fangtianxia(file_name=v)

    if method == 'build_tieba_links':
        build_tieba_links()

    if method == 'build_tieba':
        file = 'links.csv'
        build_tieba(file)
