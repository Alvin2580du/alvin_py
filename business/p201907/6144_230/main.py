from bs4 import BeautifulSoup
import requests
from tqdm import trange
import re


def get_paiming():
    fw = open('排名.txt', 'w', encoding='utf-8')

    for page in trange(1, 614):
        url = 'http://ybt.ssoier.cn:8088/ranklist.php?page={}'.format(page)
        res = requests.get(url)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')
        news = soup.findAll("tr")
        for one in news:
            rlist = one.text.split("  ")
            if len(rlist) == 5:
                fw.writelines(one.text + "\n")


def get_timu():
    fw = open('题目列表.txt', 'w', encoding='utf-8')
    fw1 = open('题目描述.txt', 'w', encoding='utf-8')

    s = '题号	题目名称	通过人数	提交总数'
    fw.writelines(s + '\n')
    for page in trange(1, 21):
        url = 'http://ybt.ssoier.cn:8088/problem_list.php?page={}'.format(page)
        res = requests.get(url)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')
        news = soup.findAll("tr")
        for one in news:
            if len(one.text.split("    ")) == 2:
                fw.writelines(one.text + "\n")

        links = soup.findAll("a", attrs={"class": "list2_link"})
        timu_text_list = []
        for link in links:
            href = "http://ybt.ssoier.cn:8088/{}".format(link['href'])
            hrefres = requests.get(href)
            hrefres.encoding = 'utf-8'
            hrefsoup = BeautifulSoup(hrefres.text, 'html.parser')
            hrefnews = hrefsoup.findAll("td", attrs={"class": "pcontent"})
            for x in hrefnews:
                res = re.search('【题目描述】.*【输入】', x.text.replace("\n", ''))
                if res:
                    timu_text = res.group().replace('【输入】', '')
                    if timu_text not in timu_text_list:
                        timu_text_list.append(timu_text)
                        fw1.writelines(timu_text.replace('\n', '').replace(' ', '')+"\n")


get_paiming()

get_timu()
