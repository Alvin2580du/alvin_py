import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
from collections import OrderedDict
import logging
import time
from tqdm import trange


def stringpro(inputs):
    inputs = str(inputs)
    return inputs.strip().replace("\n", "").replace("\t", "").lstrip().rstrip()


def isurl(url):
    if requests.get(url).status_code == 200:
        return True
    else:
        return False


def urlhelper(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
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


def coindesk():
    save = []
    for page in range(100):
        time.sleep(3)
        urls = 'https://www.coindesk.com/page/{}/'.format(page)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('div', attrs={"class": "post-info"})
            for one in range(len(resp)):
                try:
                    rows = OrderedDict()
                    post_info = resp[one]
                    timeauthor = post_info.findAll('p', attrs={'class': 'timeauthor'})
                    rows['author'] = timeauthor[0].find('cite').text
                    rows['time'] = timeauthor[0].find('time')['datetime'].replace("T", " ").split("+")[0]
                    # timeauthor = post_info.p.text
                    # rows['author'] = str(timeauthor).split("\n")[1]
                    # rows['time'] = str(timeauthor).split("\n")[0].replace("|", "")
                    title = post_info.find('h3').get_text()
                    rows['title'] = title
                    position_link = post_info.findAll('a', attrs={'class': 'fade'})
                    link = position_link[0]['href']
                    link_html = urlhelper(link)
                    link_soup = BeautifulSoup(link_html, "lxml")
                    resp2 = link_soup.findAll('p')
                    content = get_content(resp2)
                    label_html = link_soup.find("p", attrs={"class": "single-tags"}).find_all("a")
                    labels = []
                    for th in label_html:
                        label = th.get_text()
                        labels.append(label)
                    rows['labels'] = ",".join(labels)
                    rows['content'] = content
                    save.append(rows)
                    print(rows)
                except Exception as e:
                    logging.warning("------------e:{},{}".format(e, urls))
                    continue
        if page % 3 == 0:
            df = pd.DataFrame(save)
            df.to_csv("./datasets/www.coindesk.com_{}.csv".format(page), index=None)


def get_author_time(inputs):
    s = str(inputs).replace(",", "").split(" ")
    if len(s) == 6:
        author = s[:1]
        time = s[2:]
    elif len(s) == 7:
        author = s[:2]
        time = s[3:]
    else:
        return '', ''
    return ' '.join(author), " ".join(time)


def get_time(inputs):
    return inputs.replace("T", " ").replace('Z', "").split(".")[0]


def ethnews():
    save = []
    for page in trange(150):
        time.sleep(1)
        urls = 'https://www.ethnews.com/news?page={}'.format(page)
        if isurl(urls):
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('div', attrs={"class": "article-thumbnail"})
            for one in range(len(resp)):
                try:
                    rows = OrderedDict()
                    post_info = resp[one]
                    author = post_info.findAll('div', attrs={'class': 'article-thumbnail__info__etc__author'})[0].text
                    times = post_info.findAll('div', attrs={'class': 'article-thumbnail__info__etc__date'})
                    rows['author'] = author
                    rows['time'] = get_time(times[0].find('h6')['data-created-short'])
                    position_link = post_info.find_all('h2', attrs={'class': 'article-thumbnail__info__title'})
                    link_pre = position_link[0].find('a', href=True)['href']
                    link = 'https://www.ethnews.com{}'.format(link_pre)
                    link_html = urlhelper(link)
                    link_soup = BeautifulSoup(link_html, "lxml")
                    title = link_soup.findAll('div', attrs={'class': 'article-cover__inner'})[0].text
                    rows['labels'] = title
                    category = link_soup.findAll('div', attrs={'class': 'article__category'})[0].text
                    rows['labels'] = category
                    summary = link_soup.findAll('p', attrs={'class': 'article__summary'})[0].text
                    rows['summary'] = summary
                    content = link_soup.findAll('div', attrs={'class': 'article__content'})
                    content = get_content(content)
                    rows['content'] = content
                    save.append(rows)
                    print(rows)
                except Exception as e:
                    logging.warning("------------e:{},{}".format(e, urls))
                    continue
        if page % 10 == 0:
            df = pd.DataFrame(save)
            df.to_csv("./datasets/www.ethnews.com_{}.csv".format(page), index=None)


def get_coinspeaker_labels(resp):
    save = []
    for i in range(len(resp)):
        content = resp[i].text
        save.append(content)
    return ",".join(save)


def coinspeaker():
    save = []
    cates = ['https://www.coinspeaker.com/category/story-of-the-day/page',
             'https://www.coinspeaker.com/category/news/cryptocurrencies/page',
             'https://www.coinspeaker.com/category/news/fintech/page',
             'https://www.coinspeaker.com/category/news/payments-and-commerce/page',
             'https://www.coinspeaker.com/category/news/internet-of-things/page']
    for c in cates:
        for page in trange(150):
            time.sleep(1)
            urls = "{}/{}".format(c, page)
            if isurl(urls):
                html = urlhelper(urls)
                soup = BeautifulSoup(html, "lxml")
                resp = soup.findAll('div', attrs={"class": "itemBlock"})
                for one in range(len(resp)):
                    try:
                        rows = OrderedDict()
                        post_info = resp[one]
                        labels_pre = post_info.findAll('a', attrs={'class': 'categoryLabel'})
                        labels = get_coinspeaker_labels(labels_pre)
                        date = post_info.findAll('div', attrs={'class': 'newsDate'})[0].text
                        summary = post_info.findAll('div', attrs={'class': 'newsExcerpt'})[0].text
                        title_pre = post_info.findAll('div', attrs={'class': 'newsTitle'})
                        title = title_pre[0].text
                        rows['title'] = title.replace("\n", "")
                        rows['labels'] = labels.replace("\n", "")
                        rows['date'] = get_month(date.replace("\n", ""))
                        rows['summary'] = summary.replace("\n", "")

                        link = title_pre[0].find('a', href=True)['href']
                        link_html = urlhelper(link)
                        link_soup = BeautifulSoup(link_html, "lxml")
                        content = link_soup.findAll('div', attrs={'class': 'entry-content'})
                        content = get_content(content)
                        rows['content'] = content.replace("\n", "")
                        save.append(rows)
                        print(rows)
                    except Exception as e:
                        logging.warning("------------e:{},{}".format(e, urls))
                        continue

            if page % 10 == 0:
                df = pd.DataFrame(save)
                df.to_csv("./datasets/www.ethnews.com_{}.csv".format(page), index=None)


def get_month(inputs):
    md = {"January": 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
          'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, "December": 12}
    s = inputs.replace(",", "").replace("th", "").split(' ')
    return "{}-{}-{}".format(s[2], md[s[0]], s[1])


if __name__ == '__main__':
    import sys

    # method = sys.argv[1]
    method = 'coinspeaker'
    if method == 'coindesk':
        # www.coindesk.com
        coindesk()

    if method == 'ethnews':
        # www.ethnews.com
        ethnews()

    if method == 'coinspeaker':
        # www.coinspeaker.com
        coinspeaker()
