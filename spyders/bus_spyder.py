import os
from tqdm import tqdm
import urllib.request
import re


# http://life.city8090.com/chengdu/daoluming/more
def street_spyder():
    urls = 'http://life.city8090.com/chengdu/daoluming/more'
    res = urllib.request.urlopen(urls).read().decode("utf-8")
    res2str = str(res).replace(" ", "").replace("\n", "")
    print(res2str)


def gongjiao_spyder():
    urls = []
    with open("./datasets/urls", 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split('"')[1]
            urls.append(line_sp)

    for x in tqdm(urls):
        fw = open("./datasets/gongjiao/{}.txt".format(x.split("/")[-1]), 'w')
        res = urllib.request.urlopen(x).read().decode("utf-8")
        fw.writelines(res)


def help_fun(file):
    out = ""
    with open(file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            out += line.replace("\n", "").replace(" ", "")
    return out


def GetMiddleStr(content, startStr, endStr):
    startIndex = content.index(startStr)
    if startIndex >= 0:
        startIndex += len(startStr)
    endIndex = content.index(endStr)
    return content[startIndex:endIndex]


def find_gongjiaozhan():
    root = './datasets/gongjiao'
    if not os.path.exists(root):
        os.makedirs(root)
    save_dir = os.path.join(root, 'dictionary')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fw = open("./datasets/dictionary/gj.txt", 'w')
    out = []
    for file in os.listdir(root):
        file_name = os.path.join(root, file)

        res = help_fun(file_name)
        reg = '{}.+?{}'.format("返程路线", "公交线路详情页面")
        gj = re.compile(reg).search(res)
        if gj:
            gj = gj.group(0)

            gj_list = re.compile("[\u4E00-\u9FA5]{2,10}").findall(gj)
            for x in gj_list:
                if x == "返程路线":
                    continue
                if x == "公交线路详情页面":
                    continue
                out.append(x)

    for x in out:
        fw.writelines(x + "\n")


street_spyder()
