import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib.request
from urllib import error
import logging
import time
import random

logging.basicConfig(level=logging.WARNING)


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
        req.add_header("Cookie",
                       """imooc_uuid=ad059205-bc1a-4d42-b7f4-dda9b52e7ab1; imooc_isnew_ct=1552824879; zg_did=%7B%22did%22%3A%20%221698b93557e14a-0920b8da47ba5b-57b143a-240480-1698b93557f36d%22%7D; loginstate=1; apsid=VmZTIzYTM1YWVjNzFmMGVlYTllMWI5OTA3NTEzOGMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMzUxMzk1MQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADRiNDhkMmNjMGQ2YWY0MDAyNzcyYjU5M2Y3YjQ1NTAxFDuOXBQ7jlw%3DOW; mc_channel=bdqdrmjt; mc_marking=4e0b0537f151197140fed11920097988; imooc_isnew=2; Hm_lvt_f0cfcccd7b1393990c78efdeebff3968=1552824889,1554906100,1554983798; IMCDNS=0; zg_f375fe2f71e542a4b890d9a620f9fb32=%7B%22sid%22%3A%201554985736984%2C%22updated%22%3A%201554985753801%2C%22info%22%3A%201554906093124%2C%22superProperty%22%3A%20%22%7B%5C%22%E5%BA%94%E7%94%A8%E5%90%8D%E7%A7%B0%5C%22%3A%20%5C%22%E6%85%95%E8%AF%BE%E7%BD%91%E6%95%B0%E6%8D%AE%E7%BB%9F%E8%AE%A1%5C%22%2C%5C%22Platform%5C%22%3A%20%5C%22web%5C%22%7D%22%2C%22platform%22%3A%20%22%7B%7D%22%2C%22utm%22%3A%20%22%7B%7D%22%2C%22referrerDomain%22%3A%20%22%22%2C%22cuid%22%3A%20%22L5kQBofFTIQ%2C%22%2C%22zs%22%3A%200%2C%22sc%22%3A%200%7D; Hm_lpvt_f0cfcccd7b1393990c78efdeebff3968=1554985754; cvde=5caf2b7192796-10""")
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')

        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


def build():
    save = []
    for page in range(80, 150):
        # rooturl = 'https://www.imooc.com/course/coursescore/id/47?page={}'.format(page)
        rooturl = 'https://www.imooc.com/course/coursescore/id/85?page={}'.format(page)

        if not isurl(rooturl):
            continue
        print(rooturl)
        html = urlhelper(rooturl)
        soup = BeautifulSoup(html, "lxml")
        resp = soup.findAll('div', attrs={'class': 'evaluation-list'})
        try:
            for i in range(len(resp)):
                try:
                    position_link = resp[i].findAll('p')
                    for x in position_link:
                        res = x.text
                        save.append(res.replace("\n", ""))
                        print(res)
                except:
                    continue
        except:
            continue
        time.sleep(random.choice(range(4)))

    df = pd.DataFrame(save)
    df.to_csv("慕课评价.csv", index=None, mode='a')
    print(df.shape)


fw = open('data.csv', 'w', encoding='utf-8')
save = []
with open('慕课评价.csv', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()

    for line in lines:
        if len(line) < 4:
            continue
        if line not in save:
            save.append(line)
            fw.writelines(line)
