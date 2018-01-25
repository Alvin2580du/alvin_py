import urllib
import urllib.parse
from urllib.parse import urlparse
import re
import os
import urllib.request
import urllib.error


def auto_down(url, filename):
    try:
        image = urllib.request.urlretrieve(url, filename)
    except urllib.error.ContentTooShortError:
        print('Network conditions is not good.Reloading.')
        image = auto_down(url, filename)
    return image


def down_load_cvpr():
    urls = "http://openaccess.thecvf.com/CVPR2017.py"
    try:
        page = urllib.request.urlopen(urls)
        html = page.read()
        reg = r'href="(.+?\.pdf)"'
        imgre = re.compile(reg)
        pdf_list = re.findall(imgre, html)
        for pdf in pdf_list:
            target_path = "/home/dms/CVPR_2017_paper"
            pdf_url = "http://openaccess.thecvf.com/" + str(pdf)
            url_parse = urlparse(pdf_url)
            file_name = url_parse.path.split("/")[-1]
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target = os.path.join(target_path, '{}'.format(file_name))
            print("[*] Downloading paper :{}".format(pdf_url))
            auto_down(pdf_url, target)
    except:
        pass


urls = 'http://life.city8090.com/chengdu/daoluming/more'
auto_down(urls, filename='street.txt')

if __name__ == "__main__":
    urls = 'http://life.city8090.com/chengdu/daoluming/more'
    auto_down(urls, filename='street.txt')
