import urllib.request
from bs4 import BeautifulSoup
import urllib.parse
import time
import os
import re
import urllib.request
import urllib.error
import urllib
# 以上是需要导入的包

if not os.path.exists("./results"):
    os.makedirs('./results')


def download(url, user_agent='iosdevlog', num_retries=2):
    headers = {'User-agent': user_agent}
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.urlopen(request)
    try:
        html = opener.read().decode('utf-8')
    except urllib.error.URLError as e:
        print('Download error:', e.reason)
        html = None
        if num_retries > 0:
            if hasattr(e, 'code') and 500 <= e.code < 600:
                html = download(url, user_agent, num_retries-1)
    return html


# 阅读数
def crawl_views_count(jianshu_url):
    jianshu = download(jianshu_url)
    views_count = re.search(r'views_count":(\d+),', jianshu).group(1)
    return views_count


def urlOpen(url):
    # 数据抓取函数，返回取到的网页
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 "
                   "Safari/537.36")
    req.add_header("Accept", "*/*")
    req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
    data = urllib.request.urlopen(req)  # 发送请求
    html = data.read().decode("utf-8")
    return html


urls = []
fw = open('./results/results.txt', 'w', encoding='gbk')
heads = "用户ID,发表时间,文章标题,阅读量,评论量,喜欢量,打赏量\n"
fw.writelines(heads)

for page in range(1, 10):
    page_time = time.time()
    root_url = 'https://www.jianshu.com/c/V2CqjW?order_by=top&page={}'.format(page)
    html = urlOpen(root_url)
    soup = BeautifulSoup(html, "lxml")  # 解析网页
    href = soup.findAll('div', attrs={"class": "content"})
    for one in href:
        a_biaoqian = one.findAll("a", attrs={"target": "_blank"})
        title = a_biaoqian[0].text
        nickname = a_biaoqian[1].text
        try:
            comments = a_biaoqian[2].text.replace("\n", "")
        except:
            comments = 0

        spans = [str(i) for i in one.findAll("span")]
        try:
            support = re.search('<span><i class="iconfont ic-list-money"></i> \d+</span>', "".join(spans)).group()
            support = support.replace('<span><i class="iconfont ic-list-money"></i> ', "").replace('</span>', "")
        except:
            support = 0

        try:
            like = re.search('<span><i class="iconfont ic-list-like"></i> \d+</span>', "".join(spans)).group()
            like = like.replace('<span><i class="iconfont ic-list-like"></i> ', "").replace('</span>', "")
        except:
            continue
        title_link = "https://www.jianshu.com{}".format(a_biaoqian[0]['href'])
        html2 = urlOpen(title_link)
        soup2 = BeautifulSoup(html2, "lxml")
        try:
            publish = soup2.find('span', attrs={"class": "publish-time"}).text.replace("\n", "")
        except:
            publish = '没有时间'
        try:
            views = crawl_views_count(title_link)
        except:
            views = 0

        # # 爬取信息包括：用户ID、发表时间、文章标题、阅读量、评论量、喜欢量和打赏量。并最终保存到一个文件夹中。
        lines_str = "{},{},{},{},{},{}，{}".format(nickname, publish, title, views, comments, like, support).replace("\n", "")
        fw.writelines(lines_str+"\n")
        print(lines_str)
