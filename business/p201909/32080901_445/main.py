# coding = utf-8
import requests
from bs4 import BeautifulSoup
import csv
import configparser
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


class AmazonSpydr:

    def __init__(self):
        self.config_file = "config.ini"  # 配置文件名称
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file, encoding="utf-8")
        self.url_file = self.config.get("config", "url_file")
        self.save_name = self.config.get("config", "save_name")
        self.produce_type = self.config.get("config", "produce_type").split(";")
        print("已选择的产品类型：", self.produce_type)

        self.fr = open('{}'.format(self.url_file), 'r').readlines()
        self.fw = open('{}'.format(self.save_name), 'w', newline='')
        self.f_csv = csv.writer(self.fw)
        self.file_headers = ['profile_name', 'ptype', 'star', 'date', 'conent']
        self.f_csv.writerow(self.file_headers)
        self.headers = {
            "accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def Spyder(self):
        total_num = 0
        line = 0
        continue_comment = 0
        for main_url in self.fr:
            line += 1
            if line == product_lines:
                try:
                    self.response = requests.get(url=main_url.replace('\n', ''), headers=self.headers)
                    self.html = self.response.text
                    self.soup = BeautifulSoup(self.html, "lxml")
                    self.total = self.soup.findAll('a', attrs={"class": 'a-link-normal a-color-base'})
                    self.comment_url_root = self.total[0]['href']
                    self.comment_num = self.total[0].findAll('h2')[0].text.replace(" customer reviews", "")
                    self.comment_page = int(int(self.comment_num) / 10) + 2
                    print("{}\n共{}页评论".format(main_url.replace('\n', ''), self.comment_page))

                    for page in range(1, self.comment_page):
                        try:
                            comment_url = "https://www.amazon.com/" + "/".join(self.comment_url_root.split('/')[:4])
                            url_format = "/ref=cm_cr_arp_d_paging_btm_next_{}?ie=UTF8&reviewerType=all_reviews&pageNumber={}".format(
                                page, page)

                            comment_html = requests.get(url=comment_url + url_format, headers=self.headers).text
                            comment_soup = BeautifulSoup(comment_html, "lxml")

                            comment_total = comment_soup.findAll('div', attrs={"class": 'a-section celwidget'})
                            for one in comment_total:
                                try:
                                    # 用户名
                                    profile_name = one.find('span', attrs={"class": 'a-profile-name'}).text
                                    # 产品类型
                                    ptype = one.find('a', attrs={"class": 'a-size-mini a-link-normal a-color-secondary'}).text
                                    if ptype not in self.produce_type:
                                        continue_comment += 1
                                        continue
                                    # 星级
                                    star = one.find('span', attrs={"class": 'a-icon-alt'}).text.replace(" out of 5 stars", "")
                                    # 评论时间
                                    date = one.find('span', attrs={"class": 'a-size-base a-color-secondary review-date'}).text
                                    # 评论内容
                                    content = one.find('span', attrs={"class": 'a-size-base review-text review-text-content'}).text
                                    rows = [profile_name, ptype, star, date, content.replace("\n", "").replace(u'\xa0', u'')]
                                    self.f_csv.writerow(rows)  # 写入文件
                                    total_num += 1
                                    print(rows)
                                except Exception as e:
                                    print('eeeeeeeeeeeeeeeeeee1', e)
                        except Exception as e:
                            print('eeeeeeeeeeeeeeeeee2', e)
                except Exception as e:
                    print('eeeeeeeeeeeeeeeeeee3', e)

            print("共{}条评论".format(total_num))
            print("共{}条评论未选择爬取".format(continue_comment))


product_lines = int(input("请输入要爬取的产品行："))

amazonspydr = AmazonSpydr()
amazonspydr.Spyder()

