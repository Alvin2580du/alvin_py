import urllib.request
from bs4 import BeautifulSoup
from urllib import error
import urllib.parse
import logging
import urllib
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
import requests

# 画图 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
USER_AGENTS = [
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'
]

proxy_list = [
    {"http": "124.88.67.81:80"},
    {"http": "124.88.67.81:80"},
    {"http": "124.88.67.81:80"},
    {"http": "124.88.67.81:80"},
    {"http": "124.88.67.81:80"},
    {"http": "123.55.177.237:808"},
    {"http": "49.85.5.75:24748"},
    {"http": "218.91.13.2:46332"},
    {"http": "123.31.176.85:8123"},
    {"http": "218.71.161.56:80"},
    {"http": "111.231.93.66:8888"},
    {"http": "218.64.69.79:8080"},
    {"http": "140.143.48.49:1080"},
    {"http": "58.247.127.145:53281"},
    {"http": "60.13.42.146:9999"},
    {"http": "182.35.85.12:9999"},
    {"http": "58.246.3.178:53281"},
    {"http": "163.204.244.199:9999"},
    {"http": "111.231.90.122:8888"},
    {"http": "140.143.48.49:1080"},
    {"http": "58.247.127.145:53281"},
    {"http": "60.13.42.14:9999"},
    {"http": "1112.250.107.37:53281"},
    {"http": "182.35.81.145:9999"},
    {"http": "218.60.8.99:3129"},
    {"http": "182.35.84.132:9999"},
    {"http": "60.211.218.78:53281"},
    {"http": "111.231.91.104:8888"},
    {"http": "163.204.244.199:9999"},
    {"http": "60.211.218.78:53281"},
    {"http": "60.211.218.78:53281"},
]


# 爬虫的方法
def urlhelper(url):
    user_agents = USER_AGENTS[-5]
    try:
        # 随机选择一个代理
        proxy = random.choice(proxy_list)
        # 使用选择的代理构建代理处理器对象
        httpproxy_handler = urllib.request.ProxyHandler(proxy)
        opener = urllib.request.build_opener(httpproxy_handler)
        req = urllib.request.Request(url)
        # 添加请求头
        req.add_header("User-Agent", user_agents)
        req.add_header("Accept", "*/*")
        req.add_header("Accept-Language", "zh-CN,zh;q=0.9")
        # data = urllib.request.urlopen(req, timeout=50)
        data = opener.open(req, timeout=50)
        # 返回爬取到的网页
        html = data.read().decode('utf-8')
        return html
    except error.URLError as e:
        logging.warning("{}".format(e))


html = urlhelper('https://coinmarketcap.com/')
# 用BeautifulSoup解析网页
soup = BeautifulSoup(html, "lxml")
# 解析所有的a标签，并返回前20个
resp = soup.findAll('a', attrs={"class": "currency-name-container link-secondary"})[1:3]
print(len(resp))

save = []

for one in resp:
    time.sleep(random.choice(range(10)))  # 休眠几秒钟，防止反爬
    start = '20190701'  # 开始时间
    end = '20190801'  # 结束时间
    link = "https://coinmarketcap.com{}historical-data/?start={}&end={}".format(one['href'], start, end)
    print(link)
    linkhtml = urlhelper(link)
    linksoup = BeautifulSoup(linkhtml, "lxml")
    linkresp = linksoup.findAll('tr', attrs={"class": "text-right"})  # 爬取所需要的数据
    print(len(linkresp))
    for his in linkresp:
        his_td = his.find_all("td")  # 数据在tr标签里面的td标签
        rows = OrderedDict()
        rows['name'] = one['href'].split("/")[-2]
        rows['Date'] = his_td[0].text.replace(",", '')
        rows['Open'] = his_td[1].text.replace(",", '')
        rows['High'] = his_td[2].text.replace(",", '')
        rows['Low'] = his_td[3].text.replace(",", '')
        rows['Close'] = his_td[4].text.replace(",", '')
        rows['Volume'] = his_td[5].text.replace(",", '')
        rows['Market Cap'] = his_td[6].text.replace(",", '')
        save.append(rows)
        print(rows)

df = pd.DataFrame(save)
df.to_csv("data7.csv", index=None, encoding='utf-8')
print(df.shape)


df = pd.read_csv("data1.csv")

rows2 = {}
for name, y in df.groupby(by='name'):
    rows2[name] = y['Close'].values.tolist()

print(rows2)

df2 = pd.DataFrame(rows2)
df2_corr = df2.corr()  # 计算相关系数
plt.subplots(figsize=(9, 9))  # 设置画面大小
sns.heatmap(df2_corr, annot=True, vmax=1, square=True, cmap="Blues")  # 热力图
plt.savefig('Relation.png')
plt.show()
