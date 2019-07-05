import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import pandas as pd
import logging
import urllib
import os
from collections import OrderedDict
from tqdm import trange

"""
爬取慕课网上的课程信息，包括：课程链接，课程id（如果没有就设置一个），课程标题，课程图片链接，授课老师，老师介绍，课程介绍，以及价格
字段设置如下：{
  "course_url": xxx,
  "course_id": "xxx",
  "title": "xxx",
  "img_url": xxx,
  "teacher": "xxx",
  "teacher_desc": "xxx",
  "desc": "xxx",
  "source_id": "xxx",
  "price": "xxx",
}

"""

if not os.path.exists('./data'):
    os.makedirs('./data')


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


def build():
    save = []

    for pg in trange(1, 30):
        try:
            urls = "https://www.imooc.com/course/list?page={}".format(pg)
            html = urlhelper(urls)
            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('a', attrs={"class": "course-card"})
            for one in resp:
                try:
                    course_id = one['href'].split("/")[-1]
                    course_banner = one.findAll("img", attrs={"class": "course-banner lazy"})
                    img_url = course_banner[0]['data-original']
                    title = one.findAll("h3", attrs={"class": "course-card-name"})[0].text
                    price = one.findAll("span", attrs={"class": "price"})[0].text
                    course_url = "https://www.imooc.com/learn/{}".format(course_id)
                    course_html = urlhelper(course_url)
                    course_soup = BeautifulSoup(course_html, "lxml")
                    desc = course_soup.findAll('div', attrs={"class": "course-description course-wrap"})[0].text
                    teachers = course_soup.findAll('span', attrs={"class": "tit"})
                    teacher = teachers[0].text
                    teacher_url = "https://www.imooc.com{}".format(teachers[0].findAll('a')[0]['href'])
                    teacher_html = urlhelper(teacher_url)
                    teacher_soup = BeautifulSoup(teacher_html, "lxml")
                    try:
                        teacher_desc = teacher_soup.findAll('p', attrs={"class": "tea-desc"})[0].text.replace("\n", "")
                    except:
                        teacher_desc = teacher_soup.findAll('div', attrs={"class": "user-sign hide"})[0].text.replace(
                            "\n", "")

                    rows = OrderedDict({
                        "course_url": course_url,
                        "course_id": course_id,
                        "title": title,
                        "img_url": img_url,
                        "teacher": teacher,
                        "teacher_desc": teacher_desc,
                        "desc": desc,
                        "source_id": course_id,
                        "price": price,
                    })
                    save.append(rows)
                except Exception as e:
                    continue
        except Exception as e:
            continue

    df = pd.DataFrame(save)
    df.to_excel("结果.xlsx", index=None)
    print(df.shape)


def excel2txt():
    data = pd.read_excel('结果.xlsx')
    fw = open('结果.txt', 'w', encoding='utf-8')
    for x, y in data.iterrows():
        res = {
            "course_url": y['course_url'],
            "course_id": y['course_id'],
            "title": y['title'],
            "img_url": y['img_url'],
            "teacher": y['teacher'],
            "teacher_desc": y['teacher_desc'],
            "desc": y['desc'],
            "source_id": y['source_id'],
            "price": y['price'],
        }
        fw.writelines(str(res).replace('\n', '').replace(' ', '') + "\n")


build()
excel2txt()
