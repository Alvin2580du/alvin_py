import urllib.request
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd
from collections import OrderedDict
import re
import os
import json
from urllib import parse
from urllib.request import urlretrieve


# 以上是需要导入的包

# 创建一个文件夹保存封面图片
if not os.path.exists("./images"):
    os.makedirs("./images")


def urlOpen(url):
    # 数据抓取函数，返回取到的网页
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 "
                   "Safari/537.36")
    req.add_header("Accept", "*/*")
    req.add_header("Accept-Language", "zh-CN,zh;q=0.8")
    data = urllib.request.urlopen(req)  # 发送请求
    html = data.read().decode('utf-8')
    return html


class_name = ['玩具', '配饰', '饰品', '工艺品', '体育用品', '辅料',
              '服饰', '袜类', '内衣', '鞋靴', '箱包', '五金', '百货',
              '家电', '厨卫', '喜庆用品', '办公文教', '帽类', '母婴',
              '个护', '数码', '汽车用品', '建材', '机械']


class_url = 'http://101.69.178.12:15221/AdvertSystem/show_type_list.php?uppertype={}'.format(parse.quote(class_name[0]))
typeurl = json.loads(urlOpen(class_url))['common'][0]['typeurl']
html = urlOpen(typeurl)
soup = BeautifulSoup(html, "lxml")  # 解析网页
href = soup.findAll('div', attrs={"class": "menu_box"})
save = []
for one in href:
    try:
        for page in range(1, 10):
            try:
                shop_url = "https://app.yiwugo.com/search/s.htm?q={}&st=1&cpage={}&deliveryPromise=&pageSize=28&areacode=10".format(parse.quote(one.text), page)
                shop_html = json.loads(urlOpen(shop_url))
                subMarketList = shop_html['subMarketList']
                numfound = shop_html['numfound']
                print("共找到{}件商品".format(numfound))
                prslist = shop_html['prslist']
                marketNames = [i['marketName'] for i in subMarketList]
                shopID = [i['id'] for i in prslist]
                for ids in shopID:
                    try:
                        info_list = json.loads(urlOpen("https://app.yiwugo.com/product/detail2.htm?column=0&productId={}".format(ids)))
                        shopinfo = info_list['shopinfo']['shop']
                        shop_path = "images\\{}".format(shopinfo['shopId'])
                        if not os.path.exists(shop_path):
                            os.makedirs(shop_path)

                        rows = OrderedDict()
                        rows['shopId'] = shopinfo['shopId']
                        rows['smallnum'] = info_list['smallnum']
                        rows['unitprice'] = info_list['unitprice']
                        rows['companyUrl'] = shopinfo['companyUrl']
                        rows['contacter'] = shopinfo['contacter']
                        rows['email'] = shopinfo['email']
                        rows['mobile'] = shopinfo['mobile']
                        rows['telephone'] = shopinfo['telephone']
                        rows['userId'] = shopinfo['userId']
                        rows['shopName'] = shopinfo['shopName']
                        rows['weixinName'] = shopinfo['weixinName']
                        rows['factoryAddress'] = shopinfo['factoryAddress']
                        rows['marketInfo'] = shopinfo['marketInfo']
                        detail = info_list['detail']
                        productDetailVO = detail['productDetailVO']
                        rows['id'] = productDetailVO['id']
                        rows['sellType'] = productDetailVO['sellType']
                        rows['title'] = productDetailVO['title']
                        rows['introduction'] = productDetailVO['introduction']
                        rows['metric'] = productDetailVO['metric']
                        rows['saleNumber'] = productDetailVO['saleNumber']
                        rows['freight'] = productDetailVO['freight']
                        rows['freightTemplateId'] = productDetailVO['freightTemplateId']
                        sdiProductsPriceList = detail['sdiProductsPriceList']
                        sdi_price_list_all = []
                        for sdi_price in sdiProductsPriceList:
                            sdi_price_list = OrderedDict()
                            sdi_price_list['sortOrder'] = sdi_price['sortOrder']
                            sdi_price_list['startNumber'] = sdi_price['startNumber']
                            sdi_price_list['endNumber'] = sdi_price['endNumber']
                            sdi_price_list['conferPrice'] = sdi_price['conferPrice']
                            sdi_price_list['sellPrice'] = sdi_price['sellPrice']
                            sdi_price_list['vipPrice'] = sdi_price['vipPrice']
                            sdi_price_list_all.append(sdi_price_list)
                        rows['sdiProductsPriceList'] = sdi_price_list_all
                        save.append(rows)
                        print(rows)

                        sdiProductsPicList = detail['sdiProductsPicList']
                        t = 0
                        for pic in sdiProductsPicList:
                            try:
                                picture = pic['picture']
                                picture1 = pic['picture1']
                                picture2 = pic['picture2']
                                picture3 = pic['picture3']
                                pic_url = re.match("http://img1.yiwugou.com.*.jpg", picture).group()
                                pic_url1 = re.match("http://img1.yiwugou.com.*.jpg", picture1).group()
                                pic_url2 = re.match("http://img1.yiwugou.com.*.jpg", picture2).group()
                                pic_url3 = re.match("http://img1.yiwugou.com.*.jpg", picture3).group()
                                save_path = os.path.join(shop_path,  '{}_0_{}.jpg'.format(productDetailVO['id'], t))
                                save_path1 = os.path.join(shop_path,  '{}_1_{}.jpg'.format(productDetailVO['id'], t))
                                save_path2 = os.path.join(shop_path,  '{}_2_{}.jpg'.format(productDetailVO['id'], t))
                                save_path3 = os.path.join(shop_path,  '{}_3_{}.jpg'.format(productDetailVO['id'], t))
                                urlretrieve(pic_url, save_path)
                                urlretrieve(pic_url1, save_path1)
                                urlretrieve(pic_url2, save_path2)
                                urlretrieve(pic_url3, save_path3)
                                t += 1
                                print("下载", t)
                            except:
                                continue
                    except:
                        continue
            except Exception as e:
                print("118{}".format(e))
                continue
    except Exception as e:
        print("122{}".format(e))
        continue


df = pd.DataFrame(save)
