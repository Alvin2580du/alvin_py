import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import requests
from collections import OrderedDict
from tqdm import tqdm, trange
import urllib.request
from urllib import error
import logging

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
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')

        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


names = ['ziranyuyanchuli', 'jiqixuexi', 'shenduxuexi', 'rengongzhineng',
         'shujuwajue', 'suanfagongchengshi', 'jiqishijue', 'yuyinshibie',
         'tuxiangchuli']
fw = open("./datasets/lagou/job.txt", 'a+')
for name in tqdm(names):
    savedata = []
    page_number = 0
    for page in range(1, 31):

        rooturl = 'https://www.lagou.com/zhaopin/{}/{}/'.format(name, page)
        if not isurl(rooturl):
            continue
        html = urlhelper(rooturl)
        soup = BeautifulSoup(html, "lxml")
        resp = soup.findAll('div', attrs={'class': 's_position_list'})
        if len(resp) == 1:
            resp = resp[0]
            resp = resp.findAll('li', attrs={'class': 'con_list_item default_list'})
            page_number += 1
            if page_number % 5 == 0:
                print(page_number)
                # 保存到本地
                df = pd.DataFrame(savedata)
                df.to_csv("./datasets/lagou/{}_{}.csv".format(name, page_number), index=None)
                savedata = []
            for i in trange(len(resp)):
                position_link = resp[i].findAll('a', attrs={'class': 'position_link'})
                link = position_link[0]['href']
                if isurl(link):
                    htmlnext = urlhelper(link)
                    soup = BeautifulSoup(htmlnext, "lxml")
                    try:
                        # 职位描述
                        job_bt = soup.findAll('dd',
                                              attrs={'class': 'job_bt'})[0].text
                    except:
                        continue
                    try:
                        # 工作名称
                        jobname = position_link[0].find('h3').get_text()
                    except:
                        continue
                    try:
                        # 工作基本要求
                        p_bot = resp[i].findAll('div',
                                                attrs={'class': 'p_bot'})[0].text
                    except:
                        continue
                    try:
                        # 月薪
                        money = resp[i].findAll('span',
                                                attrs={'class': 'money'})[0].text
                    except:
                        continue
                    try:
                        # 行业
                        industry = resp[i].findAll('div',
                                                   attrs={'class': 'industry'})[0].text
                    except:
                        continue
                    try:
                        # 公司名字
                        company_name = resp[i].findAll(
                            'div', attrs={'class': 'company_name'})[0].text
                    except:
                        continue
                    rows = OrderedDict()
                    rows["jobname"] = jobname.replace(" ", "")
                    rows["money"] = money
                    rows["company_name"] = company_name.replace("\n", "")
                    rows["p_bot"] = p_bot.strip().replace(" ", ""). \
                        replace("\n", ",").replace("/", ",")
                    rows["industry"] = industry.strip().\
                        replace("\t", "").replace("\n", "")
                    rows["job_bt"] = job_bt
                    # for k, v in rows.items():
                    #     print(v)
                    #     fw.writelines(v+"\n")
                    savedata.append(rows)
                    print()
                    print(rows)

