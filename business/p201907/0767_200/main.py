import time
import requests
from bs4 import BeautifulSoup
import xlwt
import random
import re
import pandas as pd

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}
res1 = requests.get('http://news.sina.com.cn/', headers=headers)
res1.encoding = 'utf-8'
soup1 = BeautifulSoup(res1.text, 'html.parser')
news1 = soup1.findAll("a", attrs={"target": "_blank"})

res2 = requests.get('https://mil.news.sina.com.cn/', headers=headers)
res2.encoding = 'utf-8'
soup2 = BeautifulSoup(res2.text, 'html.parser')
news2 = soup2.findAll("a", attrs={"target": "_blank"})

news = news1 + news2

newlist = []
workbook = xlwt.Workbook(encoding='UTF-8', style_compression=0)
sheet = workbook.add_sheet('News', cell_overwrite_ok=True)
label = ['title', 'url', 'publish_time', 'source', 'keywors']
for col in range(int(len(label))):
    sheet.write(0, col, label[col])

total = 0
for new in news:
    href = re.search("http.*shtml", new['href'])
    if href:
        new_res = requests.get(href.group())
        new_res.encoding = 'utf-8'
        soup = BeautifulSoup(new_res.text, 'html.parser')
        result = soup.select(".date-source")
        publish_time = soup.select(".date-source span.date")
        source = soup.select(".date-source a")
        keywords = soup.select(".keywords")

        if len(publish_time) > 0:
            date = publish_time[0].text
        else:
            date = ''

        if len(source) > 0:
            laiyuan = source[0].text
        else:
            laiyuan = ''

        if len(keywords) > 0:
            kw = keywords[0].text.strip()
        else:
            kw = ''

        article = {"title": new.text, "url": new['href'], "publish_time": date, "source": laiyuan, "keywors": kw}
        if laiyuan:
            newlist.append(article)
            print(total, article)
            time.sleep(random.choice(range(1, 4)))
            total += 1
            if total > 301:
                break

for row, news in enumerate(newlist):
    for col, item in enumerate(news):
        sheet.write(row + 1, col, news[item])


workbook.save('News.xls')


df = pd.DataFrame(newlist)
df.to_excel("News.xlsx", index=None)
print(df.shape)
