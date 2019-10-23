import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import urllib
import pandas as pd
from collections import OrderedDict
from tqdm import trange
import time
import random


def urlhelper2(url):
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
        data = urllib.request.urlopen(req, timeout=60)

        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        print(e)


def stringPre(inputs):
    out = inputs.replace("customer reviews", "").replace("answered questions", "").replace("\n", "").replace(' ', '')
    out = out.replace("\xa0", " ").replace(",", "")

    return out


def AmazonSpyder(url, start_page, page_n=10):
    start_t = time.time()
    save = []
    for page in trange(start_page, start_page + page_n):
        page_time = time.time()
        time_cost = "time cost:{}".format("{:0.3f}".format(page_time - start_t))
        print(time_cost)
        time.sleep(random.choice(range(10)))

        try:
            url = url.split('&dc')[0] + "&dc" + "&page={}".format(page) + url.split('&dc')[1].split("&ref=")[
                0] + "&ref=sr_pg_{}".format(page)
            html = urlhelper2(url)
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            print(url)
            continue

        try:
            resp = soup.findAll('a', attrs={"class": "a-link-normal a-text-normal"})
        except Exception as e:
            print(url)
            continue

        num = 0
        for item in resp:
            time.sleep(random.choice(range(5)))
            num += 1
            if num == 1:
                continue
            try:
                href = "https://www.amazon.co.uk" + item['href']
                href_html = urlhelper2(href)
                href_soup = BeautifulSoup(href_html, "lxml")
            except Exception as e:
                continue

            try:
                alinknormal = href_soup.findAll("a", attrs={"id": "bylineInfo"})[0].text
            except Exception as e:
                alinknormal = ''

            try:
                acrCustomerReviewText = stringPre(
                    href_soup.findAll('span', attrs={"id": "acrCustomerReviewText"})[0].text)
            except Exception as e:
                acrCustomerReviewText = ''

            try:
                askATFLink = stringPre(href_soup.findAll("a", attrs={"id": "askATFLink"})[0].text)
            except Exception as e:
                askATFLink = ''
            try:
                paddingright = stringPre(href_soup.findAll('span', attrs={"class": "olp-padding-right"})[0].text)
                paddingright = paddingright.split(" ")[0]
            except Exception as e:
                paddingright = ''

            ASIN = None
            date_ = None

            try:
                wrapperGBlocale = href_soup.findAll('div', attrs={"class": "attrG"})
                for wg in wrapperGBlocale:
                    text = [i for i in wg.text.split("\n") if "ASIN" in i]
                    if text:
                        ASIN = text[0].replace("ASIN", "")
                    datefirstavailable = [i for i in wg.text.split("\n") if "Date First Available" in i]
                    if datefirstavailable:
                        date_ = datefirstavailable[0].replace("Date First Available", "")
            except Exception as e:
                print('e8', e)
                continue

            rows = OrderedDict()
            rows['alinknormal'] = alinknormal
            rows['acrCustomerReviewText'] = acrCustomerReviewText
            rows['askATFLink'] = askATFLink
            rows['paddingright'] = paddingright
            rows['datefirstavailable'] = date_
            rows['ASIN'] = ASIN
            save.append(rows)
            print(rows)

    df = pd.DataFrame(save)
    df.to_excel("爬虫结果_{}.xlsx".format(start_page), index=None)
    print(df.shape)


url = 'https://www.amazon.co.uk/s?i=diy&bbn=79903031&rh=n%3A79903031%2Cp_72%3A419154031&lo=list&dc&fst=as%3Aoff&qid' \
      '=1568119568&rnid=419152031&ref=sr_nr_p_72_2 '

AmazonSpyder(url, start_page=100, page_n=4)
