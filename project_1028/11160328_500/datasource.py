import urllib.request
from bs4 import BeautifulSoup
import os
urls = "http://www.gov.cn/guowuyuan/baogao.htm"
req = urllib.request.Request(urls)

data = urllib.request.urlopen(req)
html = data.read().decode('utf-8')
soup = BeautifulSoup(html, "lxml")
resp = soup.findAll('div', attrs={"class": "history_report"})[0]
save = []
for one in str(resp).split("</a>"):
    one_split = one.replace("\r\n", "").split('"')
    for x in one_split:
        if "http" in x:
            if x not in save:
                save.append(x)


for two in save:
    try:
        year = two.split("/")[4]
        print(year)
        if not os.path.exists('./data'):
            os.makedirs('./data')
        fw = open("./data/{}.txt".format(year), 'w', encoding='utf-8')
        req = urllib.request.Request(two)
        data = urllib.request.urlopen(req)
        html = data.read().decode('utf-8')
        soup = BeautifulSoup(html, "lxml")
        resp = soup.findAll('div', attrs={"class": "pages_content"})[0].text
        fw.writelines(resp)
    except:
        print(two)




