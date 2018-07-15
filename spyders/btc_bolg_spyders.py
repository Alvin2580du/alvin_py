import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
from collections import OrderedDict
import logging
import time
import urllib
from selenium import webdriver
from tqdm import trange
import json
from selenium.webdriver.common.keys import Keys


def stringpro(inputs):
    inputs = str(inputs)
    return inputs.strip().replace("\n", "").replace("\t", "").lstrip().rstrip()


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


def urlhelper(url):
    user_agents = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", user_agents)
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


def get_content(resp):
    save = []
    for i in range(len(resp)):
        content = resp[i].text
        save.append(stringpro(content))
    return " ".join(save)


def get_label(html):
    save = []
    if len(html) == 1:
        soup = BeautifulSoup(html[0], "lxml").find_all("a")
        for x in soup:
            save.append(x.text)
        return ",".join(save)


def get_month(inputs):
    md = {"January": 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
          'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, "December": 12}
    s = inputs.replace(",", "").replace("th", "").split(' ')
    return "{}-{}-{}".format(s[2], md[s[0]], s[1])


def get_time_huobiglobal(inputs):
    return inputs.replace("T", " ").replace('Z', "")


def huobiglobal():
    # https://huobiglobal.zendesk.com/hc/zh-cn/sections/360000007051	//火币
    urls = "https://huobiglobal.zendesk.com/hc/zh-cn/sections/360000039481-%E9%87%8D%E8%A6%81%E5%85%AC%E5%91%8A"
    print(urls)
    if isurl(urls):
        html = urlhelper(urls)
        soup = BeautifulSoup(html, "lxml")
        resp = soup.findAll('a', attrs={"class": "article-list-link"})
        save = []
        for one in range(len(resp)):
            try:
                rows = OrderedDict()
                title = resp[one].text
                rows['title'] = title
                link = "{}{}".format("https://huobiglobal.zendesk.com", resp[one]['href'])
                if isurl(link):
                    html_link = urlhelper(link)
                    soup_link = BeautifulSoup(html_link, "lxml")
                    rows['datetime'] = get_time_huobiglobal(soup_link.findAll('time')[0]['datetime'])
                    resp_link = soup_link.findAll('div', attrs={"class": "article-body"})[0].text
                    content = stringpro(resp_link)
                    rows['content'] = content
                    save.append(rows)
                    print(rows)
            except Exception as e:
                print(e)
                continue

        df = pd.DataFrame(save)
        df.to_csv("./datasets/huobiglobal.csv", index=None)
    else:
        print("链接错误")


def bithumb_cafe():
    # http://bithumb.cafe/notice
    links_tmp = []
    # browser = webdriver.Chrome()

    opt = webdriver.ChromeOptions()
    opt.set_headless()
    browser = webdriver.Chrome(options=opt)

    for page in range(1, 38):
        urls = "https://bithumb.cafe/notice/page/{}".format(page)
        browser.get(urls)
        time.sleep(5)
        links = browser.find_elements_by_xpath("//*[@href]")
        for link in links:
            try:
                rows = OrderedDict()
                link_next = link.get_attribute('href')
                if "archives" not in link_next:
                    continue
                links_tmp.append(link_next)
            except Exception as e:
                print("eeeeeeeeeeeeeeee:{}".format(e))
                continue

    print(len(links_tmp), links_tmp[:10])
    save = []

    for url in links_tmp:
        try:
            browser.get(url)
            print("正在抓取：{}".format(url))
            titles = browser.find_element_by_class_name(name='entry-title').text
            rows['title'] = titles
            author = browser.find_element_by_class_name(name='posted-author').text
            rows['author'] = author
            date = browser.find_element_by_class_name(name='posted-date').text
            rows['date'] = date
            tags = browser.find_element_by_class_name(name='posted-tags').text
            rows['tags'] = tags
            content = browser.find_element_by_class_name(name='entry-content').text
            rows['content'] = stringpro(content)
            save.append(rows)
            print(rows)
        except Exception as e:
            print("eeeeeeeeeeeeeeee:{}".format(e))
            continue
    df = pd.DataFrame(save)
    df.to_csv("./datasets/huobiglobal.csv", index=None)


def www_upbit_com():
    # browser = webdriver.Chrome()

    opt = webdriver.ChromeOptions()
    opt.set_headless()
    browser = webdriver.Chrome(options=opt)

    save = []
    home = 'https://www.upbit.com/home'
    browser.get(home)
    time.sleep(15)
    for page in range(25, 50):
        try:
            rows = OrderedDict()
            url = "https://www.upbit.com/service_center/notice?id={}".format(page)
            browser.get(url)
            content = browser.find_element_by_class_name(name='txtB').text
            title_class = browser.find_element_by_class_name(name='titB')
            title = title_class.find_element_by_tag_name('strong').text
            times_str = title_class.find_element_by_tag_name('span').text
            times = times_str.split('|')[0].split(" ")[1:]
            num = times_str.split("|")[1].split(" ")[1]
            rows['title'] = title
            rows['times'] = " ".join(times)
            rows['num'] = num
            rows['content'] = stringpro(content)
            save.append(rows)
        except Exception as e:
            continue

    df = pd.DataFrame(save)
    df.to_csv("./datasets/www_upbit_com.csv", index=None)


def bian():
    root = 'https://support.binance.com'

    save = []

    for page in trange(1, 6):
        try:
            urls = "https://support.binance.com/hc/zh-cn/sections/115000202591?page={}#articles".format(page)
            print("Spyder：{}".format(urls))
            if isurl(urls):
                html = urlhelper(urls)
                soup = BeautifulSoup(html, "lxml")
                resp = soup.findAll('li', attrs={"class": "article-list-item"})
                for i in range(len(resp)):
                    rows = OrderedDict()
                    try:
                        link_pre = "{}{}".format(root, resp[i].find('a', href=True)['href'])
                        if isurl(link_pre):
                            html_next = urlhelper(link_pre)
                            soup_next = BeautifulSoup(html_next, "lxml")
                            resp_next = soup_next.findAll('div', attrs={"class": "article-body"})[0].text
                            rows['content'] = stringpro(resp_next)
                            times = soup_next.find('time')
                            rows['times'] = get_time_huobiglobal(times['datetime'])
                            save.append(rows)
                    except Exception as e:
                        continue
        except Exception as e:
            logging.warning("page error:::::{}, {}".format(e, page))
            continue
    df1 = pd.DataFrame(save)
    df1.to_csv("./datasets/bian_新币上线.csv", index=None)

    save2 = []

    for page in trange(1, 10):
        try:
            urls = "https://support.binance.com/hc/zh-cn/sections/115000202591-%E6%9C%80%E6%96%B0%E5%85%AC%E5%91%8A?page={}#articles".format(
                page)
            print("Spyder：{}".format(urls))
            if isurl(urls):
                html = urlhelper(urls)
                soup = BeautifulSoup(html, "lxml")
                resp = soup.findAll('li', attrs={"class": "article-list-item"})
                for i in range(len(resp)):
                    rows = OrderedDict()
                    try:
                        link_pre = "{}{}".format(root, resp[i].find('a', href=True)['href'])
                        if isurl(link_pre):
                            html_next = urlhelper(link_pre)
                            soup_next = BeautifulSoup(html_next, "lxml")
                            resp_next = soup_next.findAll('div', attrs={"class": "article-body"})[0].text
                            rows['content'] = stringpro(resp_next)
                            times = soup_next.find('time')
                            rows['times'] = get_time_huobiglobal(times['datetime'])
                            save2.append(rows)
                    except Exception as e:
                        continue
        except Exception as e:
            logging.warning("page error:::::{}, {}".format(e, page))
            continue
    df2 = pd.DataFrame(save2)
    df2.to_csv("./datasets/bian_最新公告.csv", index=None)


def korbitblog_tumblr_com():
    save = []
    for page in range(1, 10):
        urls = 'http://korbitblog.tumblr.com/page/{}'.format(page)
        if isurl(urls):
            try:
                html = urlhelper(urls)
                soup = BeautifulSoup(html, "lxml")
                resp = soup.findAll('article', attrs={"class": "post post-text"})
                for i in range(len(resp)):
                    rows = OrderedDict()
                    title = resp[i].find("h1").text
                    times = resp[i].find("time")['datetime']
                    content = resp[i].find("div").text
                    rows['title'] = title.replace("\n", "")
                    rows['times'] = times
                    rows['content'] = stringpro(content)
                    save.append(rows)
                    print(rows)
            except Exception as e:
                logging.warning("{}{}".format(urls, e))
                continue

    df2 = pd.DataFrame(save)
    df2.to_csv("./datasets/korbitblog.csv", index=None)


def bitfinex_com_post():
    # https://www.bitfinex.com
    save = []
    page = 15
    for page in range(1, page + 1):
        try:
            urls = "https://www.bitfinex.com/posts?ip=124.155.55.165&page={}&user_agent=Mozilla%2F5.0+%28Windows+NT+10.0%3B+Win64%3B+x64%29+AppleWebKit%2F537.36+%28KHTML%2C+like+Gecko%29+Chrome%2F64.0.3282.186+Safari%2F537.36".format(
                page)

            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('div', attrs={"class": "section change-log-section sub_sec"})
            for i in range(len(resp)):
                try:
                    rows = OrderedDict()

                    times = resp[i].find('span', href=True).text
                    title = resp[i].find('a', href=True).text
                    content = resp[i].find_all("div")[1].text
                    rows['title'] = title.replace("\n", "")
                    rows['times'] = times
                    rows['content'] = stringpro(content)
                    save.append(rows)
                    print(rows)
                except Exception as e:
                    continue
        except Exception as e:
            logging.warning("{},{}".format(page, e))

    df2 = pd.DataFrame(save)
    df2.to_csv("./datasets/bitfinex.csv", index=None)


def bitstamp_net_v1():
    save = []
    for page in range(43):
        try:
            opt = webdriver.ChromeOptions()
            opt.set_headless()
            driver = webdriver.Chrome(options=opt)

            driver.get('https://www.bitstamp.net/ajax/news/?start={0}&limit=10'.format(page))
            driver.refresh()
            time = driver.find_elements_by_css_selector('body > article > a > hgroup > h3 > time')
            title = driver.find_elements_by_css_selector('body > article > a > section > h1 ')
            content = driver.find_elements_by_css_selector('body > article > a > section > div')
            for a in time:
                try:
                    rows = OrderedDict()
                    rows['title'] = title[time.index(a)].text.replace('\r', '').replace('\n', '')
                    rows['time'] = a.text
                    rows['content'] = stringpro(content[time.index(a)].text)
                    save.append(rows)
                except Exception as e:
                    continue

        except Exception as e:
            logging.warning("{},{}".format(page, e))
            continue

    df = pd.DataFrame(save)
    df.to_csv("./datasets/bitstamp_net.csv", index=None)


def zb_com():
    # browser = webdriver.Chrome()
    root = 'https://www.zb.com'
    opt = webdriver.ChromeOptions()
    opt.set_headless()
    browser = webdriver.Chrome(options=opt)
    browser.get(root)
    print("{}".format(root))
    links_tmp = []
    for page in trange(1, 45):
        try:
            urls = "https://www.zb.com/i/blog?page={}".format(page)
            browser.get(urls)
            time.sleep(2)
            links = browser.find_elements_by_xpath("//*[@href]")
            for link in links:
                try:
                    link_next = link.get_attribute('href')
                    if "item" not in link_next:
                        continue
                    links_tmp.append(link_next)
                except Exception as e:
                    logging.warning("eeeeeeeeeeeeeeee:{}".format(e))
                    continue
        except Exception as e:
            logging.warning("{}".format(e))
            continue
    print(len(links_tmp), links_tmp[:10])
    save = []
    for url in links_tmp:
        try:
            browser.get(url)
            print("正在抓取：{}".format(url))
            titles_selector = "body > div > div > section > div > div > div > h2.align-center"
            titles_class = browser.find_elements_by_css_selector(titles_selector)
            time_selector = 'body > div > div > section > div > div > div > p.align-center > span'
            time_class = browser.find_elements_by_css_selector(time_selector)
            content_selector = 'body > div > div > section > div > div > div > article.page-content.clearfix'
            content_class = browser.find_elements_by_css_selector(content_selector)
            for a in time_class:
                rows = OrderedDict()
                rows['title'] = titles_class[time_class.index(a)].text
                rows['time'] = time_class[time_class.index(a)].text
                rows['content'] = stringpro(content_class[time_class.index(a)].text)
                save.append(rows)
                print(rows)
        except Exception as e:
            logging.warning("{},{}".format(url, e))
            continue

    df = pd.DataFrame(save)
    df.to_csv("./datasets/www_zb_com.csv", index=None)


def medium_com_gemini(total_page=50):
    urls = "https://medium.com/gemini"
    print(urls)
    opt = webdriver.ChromeOptions()
    opt.set_headless()
    browser = webdriver.Chrome(options=opt)
    browser.get(urls)
    time.sleep(3)
    links_tmp = []

    total = 0
    links = browser.find_elements_by_xpath("//*[@href]")
    for link in links:
        try:
            link_next = link.get_attribute('href')
            if 'home' not in link_next:
                continue
            links_tmp.append(link_next)
            total += 1
        except Exception as e:
            logging.warning("eeeeeeeeeeeeeeee:{}".format(e))
            continue

    loops = 0
    loops_limit = 4
    while True:
        loops += 1
        print("have get urls number :{}".format(len(list(set(links_tmp)))))
        if len(list(set(links_tmp))) < total_page:
            js = "var q=document.documentElement.scrollTop=10000"
            browser.execute_script(js)
            browser.get(urls)
            links = browser.find_elements_by_xpath("//*[@href]")
            for link in links:
                try:
                    link_next = link.get_attribute('href')
                    if 'home' not in link_next:
                        continue
                    links_tmp.append(link_next)
                    total += 1
                except Exception as e:
                    logging.warning("eeeeeeeeeeeeeeee:{}".format(e))
                    continue
        else:
            break
        if loops > loops_limit:
            break
    links_tmp_set = list(set(links_tmp))
    print(len(links_tmp_set), links_tmp_set[-1])

    save = []
    for x in links_tmp_set:
        try:
            rows = OrderedDict()
            browser.get(x)
            title_class = browser.find_element_by_class_name(name='section-content')
            title = title_class.find_element_by_tag_name('h1').text
            rows['title'] = title
            times = browser.find_element_by_tag_name('time').get_attribute('datetime')
            rows['times'] = times.replace("T", " ").replace(".000Z", "").replace("Z", "")  # 2017-08-03T07:00:00.000Z
            contents = title_class.find_elements_by_tag_name('p')
            contents_tmp = []
            for content in contents:
                contents_tmp.append(content.text)
            content_save = stringpro("".join(contents_tmp))
            rows['content'] = content_save
            save.append(rows)
            print(rows)
        except Exception as e:
            logging.warning("{},{}".format(x, e))
            continue

    df = pd.DataFrame(save)
    df.to_csv('./datasets/medium_com_gemini.csv', index=None)


def coinone_co_kr():
    urls = 'https://coinone.co.kr/talk/notice'
    print(urls)
    save = []
    for page in trange(1, 3):
        try:
            headers = {
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'accept-encoding': 'gzip, deflate, br',
                'accept-language': 'zh-CN,zh;q=0.9',
                'cache-control': 'max-age=0',
                'cookie': '__cfduid=d0edaeb91ecd9d30041a66948b623c6ad1530108051; csrftoken=KMkLaovynntM7YwQKcjoLyoeCiBNJzWBOBvsfkPGBaC2sxTHsnoqP1HGeMbtDj86; _ga=GA1.3.201037742.1530108057; _gid=GA1.3.1255424917.1531621144; cf_clearance=54161a3d830bb06edf80d0df28137e785a0bc7de-1531667328-14400',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36',
            }
            data = requests.get(
                'https://coinone.co.kr/api/talk/notice/?page={0}&searchWord=&searchType=&ordering=-created_at'.format(page),
                headers=headers)
            soup = BeautifulSoup(data.text, 'lxml')
            contents = soup.text
            new_contents = json.loads(contents)
            rows = OrderedDict()
            results = new_contents['results']
            print(len(results))
            for result in results:
                try:
                    rows['title'] = result['title']
                    rows['card_category'] = result['card_category']
                    rows['summary'] = result['summary']
                    rows['content'] = result['sanitized_content']
                    save.append(rows)
                except Exception as e:
                    logging.warning(e)
                    continue
        except Exception as e:
            logging.warning("e:{}".format(e))
            continue

    df = pd.DataFrame(save)
    df.to_csv('./datasets/coinone_co_kr.csv', index=None)


def blog_coinbase_com(total_page=100):
    urls = 'https://blog.coinbase.com/latest'
    print(urls)
    opt = webdriver.ChromeOptions()
    opt.set_headless()
    browser = webdriver.Chrome(options=opt)
    browser.get(urls)

    links_tmp = []
    total = 0
    links = browser.find_elements_by_xpath("//*[@href]")
    for link in links:
        try:
            browser.find_element_by_xpath("//*[@href]").send_keys(Keys.DOWN)
            link_next = link.get_attribute('href')
            if len(link_next.split("/")[-1]) < 50:
                continue
            if link_next in links_tmp:
                continue
            links_tmp.append(link_next)
            total += 1
        except Exception as e:
            logging.warning("eeeeeeeeeeeeeeee:{}".format(e))
            continue

    loops = 0
    loops_limit = 10
    while True:
        loops += 1
        print("have get urls number :{}".format(len(list(set(links_tmp)))))
        if len(list(set(links_tmp))) < total_page:
            js = "var q=document.documentElement.scrollTop=100000"
            browser.execute_script(js)
            links = browser.find_elements_by_xpath("//*[@href]")
            for link in links:
                try:
                    link_next = link.get_attribute('href')
                    if len(link_next.split("/")[-1]) < 50:
                        continue
                    if link_next in links_tmp:
                        continue
                    links_tmp.append(link_next)
                    total += 1
                except Exception as e:
                    logging.warning("eeeeeeeeeeeeeeee:{}".format(e))
                    continue
        else:
            break
        if loops > loops_limit:
            break
    links_tmp_set = list(set(links_tmp))
    print(len(links_tmp_set), links_tmp_set[-1])

    save = []
    for x in links_tmp_set:
        try:
            rows = OrderedDict()
            browser.get(x)
            title_class = browser.find_element_by_class_name(name='section-content')
            title = title_class.find_element_by_tag_name('h1').text
            rows['title'] = title
            contents_class = title_class.find_elements_by_tag_name('p')
            contents_tmp = []
            for content in contents_class:
                contents_tmp.append(content.text)
            content_save = stringpro("".join(contents_tmp))
            rows['content'] = content_save
            print(rows)
            save.append(rows)

        except Exception as e:
            logging.warning("{},{}".format(x, e))
            continue

    df = pd.DataFrame(save)
    df.to_csv('./datasets/coinone_co_kr.csv', index=None)


if __name__ == '__main__':
    import sys

    # method = sys.argv[1]
    method = 'huobiglobal'
    if method == 'huobiglobal':
        huobiglobal()

    if method == 'bithumb_cafe':
        bithumb_cafe()

    if method == 'www_upbit_com':
        www_upbit_com()

    if method == 'bian':
        bian()

    if method == 'korbitblog_tumblr_com':
        korbitblog_tumblr_com()

    if method == 'bitfinex_com_post':
        bitfinex_com_post()

    if method == 'bitstamp_net_v1':
        bitstamp_net_v1()

    if method == 'zb_com':
        zb_com()

    if method == 'medium_com_gemini':
        medium_com_gemini()

    if method == 'coinone_co_kr':
        coinone_co_kr()

    if method == 'blog_coinbase_com':
        blog_coinbase_com()
