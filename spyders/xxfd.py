import urllib.request
from bs4 import BeautifulSoup
from tqdm import trange
import pandas as pd
import requests
import time

save = []
fw = open("./datasets/沣东.txt", 'a+', encoding="utf-8")
for i in trange(172):
    url = "http://www.xxfd.gov.cn/xwzx/newsList.chtml?pt=&id=ei2Abu&p={}".format(i)
    if requests.get(url).status_code != 200:
        continue
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko)"
                   " Chrome/45.0.2454.101 Safari/537.36")
    req.add_header("Accept", "*/*")
    req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
    data = urllib.request.urlopen(req)
    html = data.read().decode('utf-8')
    soup = BeautifulSoup(html, "lxml")
    position_link = soup.findAll('a', attrs={'target': '_blank'})
    try:
        for page in range(len(position_link)):
            link = "http://www.xxfd.gov.cn{}".format(position_link[page]['href'])
            data1 = urllib.request.urlopen(link).read().decode('utf-8')
            soup = BeautifulSoup(data1, "lxml")
            text = soup.find('h3').get_text()
            save.append(text)
            fw.writelines("{}".format(text)+"\n")
            print(text)
            time.sleep(3)
    except:
        pass

df = pd.Series(save)
df.to_csv("./datasets/沣东新城公告.csv", index=None)
