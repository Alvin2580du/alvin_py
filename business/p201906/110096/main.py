import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import trange

save = []
# 设置城市和对应的字母，下面构造城市的url 用
citys = {'bj': "北京", 'xa': "西安", 'cd': "成都", 'cq': "重庆", 'sh': "上海", 'sz': "深圳", 'gz': "广州",
         'hz': "杭州", 'dl': "大连", 'nj': "南京", 'sjz': "石家庄", 'sy': "沈阳", 'tj': "天津", 'wh': "武汉",
         'xm': "厦门", 'cs': '长沙', 'zz': '郑州', 'ty': '太原', 'hf': '合肥', 'fs': "佛山", 'hui': '惠州',
         'jn': '济南', 'zs': "中山"}

# 遍历所有的城市
for city in citys.keys():
    print(city)
    # 每次取100列表页
    for i in trange(100):
        try:
            # 构造链接
            url = 'https://{}.lianjia.com/ershoufang/pg{}/'.format(city, i)
            # 发送请求
            req = urllib.request.Request(url)
            # 添加请求头
            req.add_header("User-Agent",
                           "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36")
            req.add_header("Accept", "*/*")
            req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
            data = urllib.request.urlopen(req)
            # 获取网页内容
            html = data.read().decode('utf-8')
            # 用BeautifulSoup解析网页内容
            soup = BeautifulSoup(html, "lxml")
            # 获取房源信息的列表
            resp = soup.findAll('div', attrs={'class': 'info clear'})
            for i in range(len(resp)):
                try:
                    # 房源总价
                    totalPrice = resp[i].findAll('div', attrs={'class': 'totalPrice'})[0].text
                    # 房源户型 信息
                    houseInfo = resp[i].findAll('div', attrs={'class': 'houseInfo'})[0].text.split("|")[1].strip()
                    # 房源位置
                    positionInfo = resp[i].findAll('div', attrs={'class': 'positionInfo'})[0].text.split("-")[
                        1].strip()
                    # 保存到字典里面
                    rows = {"totalPrice": totalPrice, "houseInfo": houseInfo, "positionInfo": positionInfo,
                            'city': citys[city]}
                    save.append(rows)
                    print(rows)
                except:
                    continue
        except:
            continue

    if len(save) > 12000:
        break

# 最后输出到excel文件里面
df = pd.DataFrame(save)
df.to_excel("data.xlsx", index=None)  # 不要索引
print(df.shape)
