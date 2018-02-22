import urllib.request
from bs4 import BeautifulSoup
from tqdm import trange
import pandas as pd
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

citys = {'bj': "北京", 'xa': "西安", 'cd': "成都", 'cq': "重庆", 'sh': "上海", 'sz': "深圳", 'gz': "广州",
          'hz': "杭州", 'dl': "大连", 'nj': "南京", 'sjz': "石家庄", 'sy': "沈阳", 'tj': "天津", 'wh': "武汉",
          'xm': "厦门", 'cs': '长沙', 'zz': '郑州', 'ty': '太原', 'hf': '合肥', 'fs': "佛山", 'hui': '惠州',
          'jn': '济南', 'zs': "中山"}

print(len(citys))
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


def xaspyder():
    for city in citys.keys():
        save_data = []
        for i in trange(120):
            url = "https://{}.fang.lianjia.com/loupan/pg{}/".format(city, i)
            print(requests.get(url).status_code)
            if requests.get(url).status_code != 200:
                continue
            req = urllib.request.Request(url)
            req.add_header("User-Agent",
                           "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
            req.add_header("Accept", "*/*")
            req.add_header("Accept-Language", "zh-CN,zh;q=0.8")

            data = urllib.request.urlopen(req)
            html = data.read().decode('utf-8')

            soup = BeautifulSoup(html, "lxml")
            resp = soup.findAll('ul', attrs={'class': 'resblock-list-wrapper'})
            resp = resp[0]
            resp = resp.findAll('li', attrs={'class': 'resblock-list'})
            for i in range(len(resp)):
                housenames = resp[i].findAll('div', attrs={'class': 'resblock-name'})
                housename = housenames[0].findAll('a', attrs={'target': '_blank'})[0].text
                herf = housenames.get('herf')
                print(herf)
                exit(1)
                try:
                    housenames = resp[i].findAll('div', attrs={'class': 'resblock-name'})
                    housename = housenames[0].findAll('a', attrs={'target': '_blank'})[0].text
                    herf = housenames.get('herf')
                    print(herf)
                    resblocktype = housenames[0].findAll('span', attrs={'class': 'resblock-type'})[0].text
                    salestatus = housenames[0].findAll('span', attrs={'class': 'sale-status'})[0].text
                except:
                    continue
                try:
                    resblocklocation = resp[i].findAll('div', attrs={'class': 'resblock-location'})
                    addressinfolist = resblocklocation[0].text.replace("\n", "")
                    quyu, address, addressinfo = addressinfolist.split("/")[0], addressinfolist.split("/")[1], \
                                                 addressinfolist.split("/")[2]
                except:
                    continue
                try:
                    resblockroom = resp[i].findAll('a', attrs={'class': 'resblock-room', 'target': '_blank'})[
                        0].text.replace("\n", "")
                except:
                    continue
                try:
                    resblockarea = resp[i].findAll('div', attrs={'class': 'resblock-area'})[0].text.replace("\n",
                                                                                                            "").replace(
                        "建面 ", "")
                except:
                    continue
                try:
                    resblockprice = resp[i].findAll('div', attrs={'class': 'main-price'})[0]
                    priceinfo = resblockprice.findAll('span', attrs={'class': 'number'})[0].text
                except:
                    continue
                try:
                    secondprice = resp[i].findAll('div', attrs={'class': 'second'})[0].text.replace("总价", "").replace(
                        "万/套起", "")
                except:
                    continue
                rows = {'housename': housename, 'resblocktype': resblocktype, 'salestatus': salestatus,
                        'address': address,
                        'addressinfo': addressinfo, 'resblockroom': resblockroom, 'resblockarea': resblockarea,
                        'priceinfo': priceinfo, 'secondprice': secondprice}
                save_data.append(rows)

        df = pd.DataFrame(save_data)
        df.to_csv("./datasets/{}.csv".format(city), line_terminator="\n", index=None, encoding='utf-8')


def draw_bar(labels, quants):
    width = 0.5
    ind = np.linspace(1, 23, 23)
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    ax.bar(ind - width / 2, quants, width, color='green')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.set_ylabel('新房均价')
    ax.set_title('全国23个大城市新房平均价格', bbox={'facecolor': '0.8', 'pad': 5})
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.savefig("bar.jpg")
    plt.close()


def analyse():
    path = './datasets'
    mean_price = []
    for file in os.listdir(path):
        filename = os.path.join(path, file)
        data = pd.read_csv(filename, usecols=['priceinfo']).values

        prices = []
        for x in data:
            if str(x[0]).isdigit():
                prices.append(int(x[0]))

        pricesmean = np.mean(prices)
        rows = {'city': citys[file.split(".")[0]], 'meanprice': pricesmean}
        mean_price.append(rows)
    df = pd.DataFrame(mean_price)
    df = df.sort_values(by='meanprice', ascending=False)
    labels = df['city']
    price = df['meanprice']
    draw_bar(labels, price)


if __name__ == "__main__":
    method = 'analyse'

    if method == 'spyder':
        xaspyder()
    if method == 'analyse':
        analyse()
