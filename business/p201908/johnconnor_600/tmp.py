import requests, threading, datetime
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ip清洗
def checkip(targeturl, ip):
    proxies = {"http": "http://" + ip, "https": "http://" + ip}  # 代理ip
    try:
        response = requests.get(url=targeturl, verify=False, proxies=proxies, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'},
                                timeout=5).status_code
        if response == 200:
            return True
        else:
            return False
    except:
        return False


# 免费代理 XiciDaili
def findip(type, pagenum, targeturl):  # ip类型,页码,目标url,存放ip的路径
    list = {'1': 'http://www.xicidaili.com/nt/',  # xicidaili国内普通代理
            '2': 'http://www.xicidaili.com/nn/',  # xicidaili国内高匿代理
            '3': 'http://www.xicidaili.com/wn/',  # xicidaili国内https代理
            '4': 'http://www.xicidaili.com/wt/'}  # xicidaili国外http代理
    url = list[str(type)] + str(pagenum)  # 配置url
    html = requests.get(url=url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'},
                        timeout=5).text
    soup = BeautifulSoup(html, 'lxml')
    all = soup.find_all('tr', class_='odd')
    for i in all:
        t = i.find_all('td')
        ip = t[1].text + ':' + t[2].text
        is_avail = checkip(targeturl, ip)
        if is_avail == True:
            # write(path=path,text=ip)
            print(ip)


def getip(targeturl):
    # truncatefile(path) # 爬取前清空文档
    start = datetime.datetime.now()  # 开始时间
    threads = []
    for type in range(4):
        for pagenum in range(10):
            t = threading.Thread(target=findip, args=(type + 1, pagenum + 1, targeturl))
            threads.append(t)
    print('开始爬取代理ip')

    for s in threads:  # 开启多线程爬取
        s.start()

    for e in threads:  # 等待所有线程结束
        e.join()
    print('爬取完成')


if __name__ == '__main__':
    targeturl = 'https://blog.csdn.net/u011928550/article/details/60154191'  # 验证ip有效性的指定url
    getip(targeturl)
