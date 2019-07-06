import urllib.request
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd
from collections import OrderedDict
import re
import os
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# 以上是需要导入的包

# 创建一个文件夹保存封面图片
if not os.path.exists("./images"):
    os.makedirs("./images")


def urlOpen(url):
    # 数据抓取函数，返回取到的网页
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 "
                   "Safari/537.36")
    req.add_header("Accept", "*/*")
    req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
    data = urllib.request.urlopen(req)  # 发送请求
    html = data.read()
    return html


# 爬虫第一步先获取所有出版社下面的书籍的链接，并保存到文件book_urls.xlsx
def get_books():
    # W开头的出版社
    chubanshes = ['外国文学出版社',
                  '外文出版社',
                  '外语教学与研究出版社',
                  '万方数据电子出版社',
                  '万国学术出版社',
                  '万卷出版公司',
                  '未来出版社',
                  '文化艺术出版社',
                  '文汇出版社',
                  '文津出版社',
                  '文物出版社',
                  '文心出版社',
                  '五洲传播出版社',
                  '五洲传播电子出版社',
                  '武汉测绘科技大学出版社',
                  '武汉出版社',
                  '武汉大学出版社',
                  '武汉理工大学出版社',
                  '武汉水利电力大学出版社',
                  '武警音像出版社']
    book_urls = []
    url = 'http://www.bookschina.com/books/publish/'
    html = urlOpen(url)
    soup = BeautifulSoup(html, "lxml")  # 解析网页
    href = soup.findAll('a', attrs={"target": "_blank"})  # 选择所有的a标签
    for i in href:
        if i.text in chubanshes:
            print("正在抓取：{}".format(i.text))
            link = "http://www.bookschina.com{}".format(i['href'])
            number = link.split("/")[-2]
            for p in range(100):
                try:
                    # 获取某一个出版社下面所有的图书，遍历全部列表页
                    page = "http://www.bookschina.com/publish/{}_AA_1_1_0_{}/".format(number, p)
                    html = urlOpen(page)
                    soup = BeautifulSoup(html, "lxml")
                    book_href = soup.findAll('div', attrs={"class": "cover"})
                    for book in book_href:
                        try:
                            # 书籍的链接在a标签里面的href属性
                            book_href = book.find('a', attrs={"target": "_blank"})['href']
                            book_url = "http://www.bookschina.com{}".format(book_href)
                            book_urls.append(book_url)
                        except Exception as e:
                            continue
                except:
                    continue
    # 保存到文件
    df = pd.DataFrame(book_urls)
    df.to_excel("book_urls.xlsx", index=None)
    print(df.shape)


# 爬虫第二步， 对第一步获取到的所有书籍链接进行获取相关信息
def save_data():
    save = []
    if os.path.exists("PublisherBooks.db"):
        os.remove('PublisherBooks.db')
        print("已删除")

    if os.path.exists("brief.txt"):
        os.remove('brief.txt')
        print("已删除")

    # 保存到数据库
    conn = sqlite3.connect('PublisherBooks.db')
    cursor = conn.cursor()
    # 创建表
    try:
        cursor.execute('create table if not exists PublisherBooks (id int primary key, xinmin char, name varchar('
                       '200), author varchar(200), publisher_info varchar(200), series_info varchar('
                       '200), otherInfor_info varchar(200), sort_info varchar(200), isbn varchar(200), '
                       'txm varchar(200), specialist varchar(200), brief varchar(200))')
        conn.commit()
    except:
        pass

    data = pd.read_excel("book_urls.xlsx").drop_duplicates()

    num = 0
    total = 23449  # 这里可以设置爬取的书籍数量, 总共23449
    for x, y in data.head(total).iterrows():
        num += 1
        print(num, y[0])
        html = urlOpen(y[0])
        soup = BeautifulSoup(html, "html.parser", from_encoding='GBK')
        try:
            #  书名字
            h1_info = soup.find('h1').text.replace("\n", "")
        except:
            h1_info = ''
        try:
            # 封面图片
            image = soup.findAll('input')
            for i in image:
                if 'http' in i['value'] and "jpg" in i['value']:
                    # 下载封面图片
                    urllib.request.urlretrieve(i['value'], filename='./images/{}.jpg'.format(h1_info))
        except:
            pass
        try:
            # 作者
            author_info = soup.find('div', attrs={"class": "author"}).text
        except:
            continue
        try:
            # 出版社
            publisher_info = soup.find('div', attrs={"class": "publisher"}).text
        except:
            publisher_info = ''

        try:
            # 所属丛书
            series_info = soup.find('div', attrs={"class": "series"}).text.replace("\n", "").replace(" ", "")
        except:
            series_info = ""
        # 其他信息
        try:
            otherInfor_info = soup.find('div', attrs={"class": "otherInfor"}).text.replace(" ", "")
            if len(otherInfor_info) < 3:
                otherInfor_info = ""
        except:
            otherInfor_info = ""
        try:
            #  排行榜
            sort_info = soup.find('div', attrs={"class": "sort"}).text
        except:
            sort_info = ''
        try:
            copyrightInfor = soup.find('div', attrs={"class": "copyrightInfor"})
            # ISBN码和条形码
            isbn = re.search('<li>ISBN：.*</li>', str(copyrightInfor)).group().replace("</li>", "").replace("<li>", "")
            txm = re.search('条形码：.*', str(copyrightInfor)).group().replace("</li>", "").replace(";", "").replace(" ",
                                                                                                                 "")
        except:
            isbn = ''
            txm = ''

        try:
            # 特色
            specialist = soup.find('div', attrs={"id": "specialist"}).text.replace("\n", "").replace(" ", "")
        except:
            specialist = ''
        # 简介
        try:
            brief = soup.find('div', attrs={"id": "brief"}).text.replace("\n", "").replace(" ", "")
        except:
            brief = ""

        # 保存到有序字典
        rows = OrderedDict()
        rows['name'] = h1_info
        rows['author'] = author_info.replace("\n", "")
        rows['publisher_info'] = publisher_info.replace("\n", "")
        rows['series_info'] = series_info.replace("\n", "")
        rows['otherInfor_info'] = otherInfor_info.replace("\n", "")
        rows['sort_info'] = sort_info.replace("\n", "")
        rows['isbn'] = isbn.replace("\n", "")
        rows['txm'] = txm.replace("\n", "")
        rows['specialist'] = specialist
        rows['brief'] = brief
        fw = open('brief.txt', 'a+', encoding='utf-8')
        print("brief", brief)
        try:
            fw.writelines("{}\n".format(brief))
        except:
            fw.writelines("\n")

        # # 保存到列表里面
        save.append(rows)

        xm = "金毓"  # 修改名字
        sqls = "insert  into  PublisherBooks(id, xinmin,name,author,publisher_info,series_info,otherInfor_info," \
               "sort_info,isbn,txm,specialist,brief) VALUES('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(
            num, xm, h1_info,
            author_info.replace("\n", ""),
            publisher_info.replace("\n", ""),
            series_info.replace("\n", ""),
            otherInfor_info.replace("\n", ""),
            sort_info.replace("\n", ""),
            isbn.replace("\n", ""),
            txm.replace("\n", ""),
            specialist,
            brief)
        cursor.execute(sqls)
        conn.commit()

    conn.close()
    # 利用pandas 保存到文件
    df = pd.DataFrame(save)
    df.to_excel("结果文件.xlsx", index=None)


def db_test():
    # 数据库测试
    conn = sqlite3.connect("PublisherBooks.db")
    c = conn.cursor()
    ret = c.execute("select * from PublisherBooks").fetchall()
    for ii in ret:
        print(ii)
    conn.close()


# get_books()
# save_data()
# db_test()
