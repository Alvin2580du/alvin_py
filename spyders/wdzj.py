"""
Author : Alvin2580du
time : 2017-12-08
网贷之家爬虫脚本
"""
import requests
import time
import random
import codecs
import filecmp
import re
import os
import pandas as pd

import jieba

from gensim.models import Word2Vec


def get_all_url():
    # 创建list, 存放所有准备下载的url链接
    url_list = []
    start_page = random.choice(range(1, 200000))
    for i in range(start_page, 5000000):
        urls = "https://bbs.wdzj.com/thread-{}-1-1.html".format(i)
        url_list.append(urls)
    return url_list


def wdzj_spyder(target_url):
    # 爬虫脚本主程序
    out = []
    heafers_list = []
    for i in range(10, 50):
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/{}.0'.format(i)}
        heafers_list.append(headers)

    headers = random.choice(heafers_list)
    proxie = {'http': 'http://122.193.14.102:80'}

    try:
        res = requests.get(url=target_url,
                           headers=headers,
                           timeout=1,
                           proxies=proxie,
                           stream=True,
                           verify=False,
                           params={'ip': '8.8.8.8'},
                           )
        res.encoding = 'utf-8'
        if res.status_code == 200:
            res_text = res.text
            out.append(res_text)
    except Exception as e:
        pass
    return out


def build_wdzj_spyder(sleep_time=3600, out_path="./download_raw_data"):
    # build
    url_list = get_all_url()
    k = 1

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for url in url_list:
        res = wdzj_spyder(url)
        if res:
            url_name = url.split("/")[-1].split(".")[0]
            out_name = os.path.join(out_path, url_name + "{}".format(".txt"))
            if os.path.isfile(out_name):
                os.remove(out_name)
            out = codecs.open(out_name, 'w', encoding='utf-8')
            print("正在下载：{} \n 文件名：{}".format(url, out_name))
            for lin in res:
                str(lin).encode(encoding='utf-8')
                out.writelines(lin)
                k += 1
                time.sleep(sleep_time)
        else:
            continue


def get_bbs(in_path="./download_raw_data", out_path="./bbs_datasets"):
    # 获得bbs的内容，并保存到文件, 获取start和end之间的内容
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files = os.listdir(in_path)
    start = "bbs-post-left"
    end = "bbs-post-right"
    p = re.compile("{}.+?{}".format(start, end))

    k = 0
    for file in files:
        save_as_file = open(os.path.join(out_path, file), 'w')

        with open(os.path.join(in_path, file), 'r') as fr:
            lines = fr.readlines()
            new_lines = str(lines).strip("\n").replace(" ", "").replace("\t", "")
            try:
                res = p.search(new_lines)
                if res:
                    out = res.group()
                    ds = "[A-Za-z0-9\-\"\+\……\（\）\一\_\、\？\～\；\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\,\[\]\.\<\>\/\?\~\！\#\\\&\*\%]"
                    r = re.sub(ds, "", out)
                    out_res = str(r).replace(" ", "").replace("\n", "").strip().lstrip().rstrip()
                    if len(out_res) > 50:
                        save_as_file.writelines(out_res)
                else:
                    continue
            except Exception as e:
                pass
        k += 1
        if k % 100 == 1:
            print("process {} file".format(k))


def make_subdatasets(size=50, old_path="./bbs_datasets"):
    #  删除空文件
    files = os.listdir(old_path)
    counter = 1
    limit = 50000
    for file in files:
        key = os.path.isfile(os.path.join(old_path, file))
        if key:
            if os.path.getsize(os.path.join(old_path, file)) < size:
                os.remove(os.path.join(old_path, file))
                counter += 1

                if counter % 100 == 1:
                    print("file:{}".format(file))

                if counter > limit:
                    break
            else:
                continue
        else:
            print("file:{}".format(file))


def is_same_file(file_path="./bbs_datasets"):
    # 判断两个文件是否一样， 如果一样， 删除其中一个
    files = os.listdir(file_path)
    length = len(files)
    common_files = []
    for i in range(length):
        for j in range(i + 1, length):
            one = os.path.join(file_path, files[i])
            two = os.path.join(file_path, files[j])
            res = filecmp.cmp(one, two)
            if res:
                common_files.append(one)
            else:
                continue
    for cf in common_files:
        os.remove(cf)
        print("{} has been removed".format(cf))


def word_cut(in_path="./bbs_datasets", out_path="./bbs_word_cut", stop_word='./dictionary/stopwords.txt'):
    # 中文分词主程序
    jieba.load_userdict("./dictionary/userdict.txt")

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files = os.listdir(in_path)
    # cut_out = open(os.path.join(out_path, "cut_all"), 'a+')

    cut_all = None
    k = 1
    for file in files:
        single_cut = []
        with open(os.path.join(in_path, file), 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                line_cut = jieba.cut(line)
                stopwords = {}.fromkeys([line.rstrip() for line in open(stop_word, 'r')])
                for seg in line_cut:
                    if seg not in stopwords:
                        single_cut.append(seg)
        k += 1
        if k % 100 == 1:
            print("{}".format(k))

        df = pd.DataFrame(single_cut).T  # 注意可能需要转置
        out_name = os.path.join(out_path, "{}.csv".format(file.split(".")[0]))
        df.to_csv(out_name, index=None, header=None)
        k += 1

        # TODO: 合并csv
        # if cut_all is None:
        #     cut_all = single_cut
        # else:
        #     cut_all = np.concatenate(np.array(cut_all), np.ndarray(single_cut))
        #     # ("./results_cut/cut_all.csv", index=None, header=None)


def merge_csv2oneText(in_path="./bbs_word_cut", out_path="./cut_bbs_all", out_name="all_data_pos.txt"):
    # 合并分词之后的文件到一个文件
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out = open(os.path.join(out_path, out_name), 'a+')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_list = os.listdir(in_path)
    for i in range(len(file_list)):
        fr = open(os.path.join(in_path, file_list[i]), 'r')
        lines = fr.readline()
        out.writelines(lines.strip() + "\n")


class TextLoader(object):
    # 读取合并后的文件，创建一个迭代器
    def __init__(self, file_name='./bbs_all/all_data_pos.txt'):
        self.path = file_name

    def __iter__(self):
        input = open(self.path, 'r')
        line = str(input.readline())
        while line is not None and len(line) > 4:
            segments = line.split(',')
            yield segments
            line = str(input.readline())


def build_embedding(file_name='./cut_bbs_all/all_data_pos.txt', out_path="embedding"):
    # word2vector 主程序
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    sentences = TextLoader(file_name)
    model = Word2Vec(sentences, size=128, window=5, min_count=2, workers=4)
    model.init_sims(replace=True)
    embedding_name = os.path.join(out_path, "hi_embedding.txt")
    embedding_bin_name = os.path.join(out_path, "hi_embedding.bin")
    model.wv.save_word2vec_format(embedding_name, binary=False)
    model.save(embedding_bin_name)


if __name__ == "__main__":
    print("")
    # 1. 先运行build_wdzj_spyder直接把网页下载下来
    # time_list = range(10, 100, 10)
    # sleep_time = random.choice(time_list)
    # build_wdzj_spyder(sleep_time=sleep_time)
    # 2. 然后去掉不想要的，保留帖子的内容
    # get_bbs()

    # 2.1 删除相同的文件，也可以不做
    # is_same_file()

    # 3. 删除空文件
    # make_subdatasets()
    # 4. 分词
    # word_cut()

    # 5. 合并分词后的文件到一个text
    # merge_csv2oneText(in_path="./bbs_datasets", out_name="bbs_raw_data.txt")

    # 需要标注数据后执行
    # 6.  执行word2vector
    # build_embedding()