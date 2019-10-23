import requests
from sklearn.cluster import DBSCAN

from bs4 import BeautifulSoup
headers = {
            "accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

for page in range(500):
    url = "https://www.amazon.co.uk/s?i=diy&bbn=79903031&rh=n%3A79903031%2Cp_72%3A419154031&lo=list&dc&page=3&fst=as%3Aoff&qid=1568119687&rnid=419152031&ref=sr_pg_3".format(page, page)
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "lxml")
    total = soup.findAll('div', attrs={"class": 'rush-component'})
    print(url)
    print(len(total))


