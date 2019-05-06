from copyheaders import headers_raw_to_dict
from bs4 import BeautifulSoup
import requests
import time
import re
import random

######### 爬虫 ##############
headers = b"""
accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8
accept-encoding:gzip, deflate, br
accept-language:zh-CN,zh;q=0.9
cache-control:max-age=0
cookie:_T_WM=1ad1436888fe8b89c7a2863f7986d7b9; SUB=_2A25xpfVRDeRhGeRI6VMT8C3FzT-IHXVTaZsZrDV6PUJbkdAKLWqmkW1NUtUcliAU3-z50ac51gSoCPiLVoblnWq1; SUHB=0wn0NF3PkyCp1I; SCF=Anw_ufU9Bwimf_jyIpvjZPuSWOGgMSGxwWbMWbz6_lrc9w18Uz2_qdur67eB2PdHI7SHFiCLhZYDCtIgm7ehNcc.
upgrade-insecure-requests:1
user-agent:Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36
"""

# 将请求头字符串转化为字典
headers = headers_raw_to_dict(headers)
for i in range(1, 3000):
    try:
        if i % 50 == 1:
            print("正在抓取第{}页".format(i))
        time.sleep(random.choice(range(10)))
        # 请求网址， 替换网址， 进行爬虫
        url = 'https://weibo.cn/comment/H8lR49Ezg?ckAll=1&oid=4361070754067168&page=' + str(i)
        # 刘强东事件
        # https://weibo.cn/comment/H8lR49Ezg?ckAll=1&oid=4361070754067168&page=
        # 奔驰维权事件
        # https://weibo.cn/comment/HpppvdIeR?uid=2656274875&rl=1&page=
        # 翟天临事件
        # https://weibo.cn/comment/Hevk0ci8R?uid=1343887012&rl=1&page=
        # https://weibo.cn/comment/Hf9C354j6?uid=1343887012&rl=0&page=
        ## https://weibo.cn/comment/Hevk0ci8R?uid=1343887012&rl=1&page=
        response = requests.get(url=url, headers=headers)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        # 评论信息
        result_1 = soup.find_all(class_='ctt')
        # 点赞数
        result_2 = soup.find_all(class_='cc')
        # 评论时间
        result_3 = soup.find_all(class_='ct')
        # 获取用户名
        result_4 = re.findall('id="C_.*?href="/.*?">(.*?)</a>', html)
        try:
            for j in range(len(result_1)):
                # 获取点赞数
                res = re.findall(r'(\d+)', result_2[j * 2].get_text())
                if len(res) > 0:
                    praise = res[0]
                    name = result_4[j]
                    text = result_1[j].get_text().replace(',', '，')
                    date = result_3[j].get_text().split(' ')[0]
                    if '@' in text:
                        if ':' in text:
                            # 去除@及用户信息
                            comment = text.split(':')[-1]
                            # 写入csv
                            with open('liuqiangdong2500.csv', 'a+') as f:
                                save = name + ',' + comment + ',' + praise + ',' + date
                                f.writelines(save + "\n")
                        else:
                            # 无评论信息时
                            with open('tmp.csv', 'a+') as f:
                                save = name + ',' + '无' + ',' + praise + ',' + date
                                f.writelines(save + "\n")
                    else:
                        # 只有评论信息
                        # 写入csv
                        with open('liuqiangdong2500.csv', 'a+') as f:
                            save = name + ',' + text + ',' + praise + ',' + date
                            f.writelines(save + "\n")

                else:
                    pass
        # 出现字符编码报错
        except:
            continue
    except:
        continue
