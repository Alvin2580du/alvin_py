import pandas as pd
from collections import OrderedDict

from pyduyp.utils.utils import is_chinese


def getnews(msg):
    out = []
    for x in msg:
        if is_chinese(x):
            out.append(x)
    return "".join(out)


def readexample():
    save_data = []
    with open("shouhu_for_text/datasets/News_info_train_example100.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            linesp = line.split("\t")
            newsid, imagesid, msg = linesp[0], linesp[2], linesp[1]

            rows = OrderedDict({"mewsid": newsid, "imagesid": imagesid, 'msg': getnews(msg)})
            save_data.append(rows)
    df = pd.DataFrame(save_data)
    df.to_csv("shouhu_for_text/datasets/News_info_train_example100.csv", index=None, line_terminator='\n', encoding="utf-8")


def News_pic_label_train():
    save_data = []
    with open("shouhu_for_text/datasets/News_pic_label_train_example100.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            linesp = line.split("\t")
            newsid, label, imagesid, msg = linesp[0], linesp[1], linesp[2], linesp[3:]

            rows = OrderedDict({"mewsid": newsid, "label": label, "imagesid": imagesid, 'msg': msg[0].replace("\n", "")})
            save_data.append(rows)
    df = pd.DataFrame(save_data)
    df.to_csv("shouhu_for_text/datasets/News_pic_label_train_example100.csv", index=None, line_terminator='\n', encoding="utf-8")
