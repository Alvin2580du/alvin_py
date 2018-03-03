import os
import pandas as pd
import re
import jieba
import matplotlib.pyplot as plt
import wordcloud
from collections import OrderedDict
from tqdm import tqdm

from scipy.misc import imread
sw = []
jieba.load_userdict("./dictionary/ai.dict")


def linesplit_bysymbol(line):
    # 先按句子切割，然后去标点,
    line = line.replace("\n", "").replace("\r", '')
    out = []
    juzi = r"[\］\［\,\！\】\【\：\，\。\?\？\)\(\(\『\』\<\>\、\；\．\[\]\（\）\〔\〕\+\和\的\与\在]"
    p = r"[\^\$\]\/\.\’\~\#\￥\#\&\*\%\”\“\]\[\&\×\@\]\"]"

    salary_pattern = r'\d+\k\-\d+\k'
    salarys = re.compile(salary_pattern).findall(line)

    for salary in salarys:
        out.append(salary)
    linesplit = re.split(juzi, line)
    for x in linesplit:

        if str(x).isnumeric():
            continue
        if len(x) < 1:
            continue
        if x == '职位描述':
            continue
        if x == '岗位要求':
            continue
        if x == '岗位职责':
            continue
        if x == '工作职责':
            continue
        if x == '岗位说明':
            continue
        res = re.sub(p, "", x)
        rescut = jieba.lcut(res)
        for w in rescut:
            if str(w).isdigit():
                continue
            out.append(w)
    return " ".join(out)


def analysis(job='suanfagongchengshi'):
    path = './datasets/lagou'
    res = []
    for file in tqdm(os.listdir(path)):
        if "{}".format(job) in file:
            file_name = os.path.join(path, file)
            data = pd.read_csv(file_name, usecols=['job_bt', 'company_name']).values
            for x in data:
                rows = OrderedDict()
                rows['company'] = x[0]
                rows['bt'] = linesplit_bysymbol(x[1])
                res.append(rows)
    df = pd.DataFrame(res)
    df.to_csv("./datasets/lagou/{}.csv".format(job), index=None)


def plot_word_cloud(file_name, savename):
    text = open(file_name, 'r', encoding='utf-8').read()

    alice_coloring = imread("suanfa.jpg")
    wc = wordcloud.WordCloud(background_color="white", width=918, height=978,
                             max_font_size=50,
                             mask=alice_coloring,
                             random_state=1,
                             max_words=80,
                             mode='RGBA',
                             font_path='msyh.ttf')
    wc.generate(text)
    image_colors = wordcloud.ImageColorGenerator(alice_coloring)

    plt.axis("off")
    plt.figure()
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis("off")
    plt.figure(dpi=600)
    plt.axis("off")
    wc.to_file(savename)


def adddict():
    with open("dictionary/ai.dict", 'a+', encoding='utf-8') as fw:
        data = pd.read_csv("./datasets/lagou/nlp.csv", usecols=['company']).values
        for x in data:
            print(x[0])
            fw.writelines(x[0] + "\n")

# analysis(job='suanfagongchengshi')
# plot_word_cloud(file_name="./datasets/lagou/suanfagongchengshi.csv", savename='suanfagongchengshi.png')
save = []
with open("dictionary/ai.dict", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        if line not in save:
            save.append(line.strip())

fw = open("dictionary/ai_dict.txt", 'w', encoding='utf-8')
for x in save:
    fw.writelines(x+"\n")

