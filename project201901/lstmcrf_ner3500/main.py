"""
第三章模型训练（实际实验内容和过程）
4.1 数据准备（pdf转txt，提取相关数据，以下有些）
4.2 序列标注（由我手工完成，标注特征词和情感标记）
4.3 模型训练的选择（基础算法选取，可以svm？或者LSTM与下面的双向BiLSTM做比对？这个对比算法您也可以适当给推荐）
4.4 模型训练的优化（优化选法选取，拟采用主流的BiLSTM+CRF，包括一次调优，预期prf测试值到90%）
4.5 测试结果展示（表格对比结果：基础算法，调优前的优化算法结果，以及调优后的优化算法结果与人工标注做对比，同时还包括模型自带的准确率）
4.6 训练结果分析（基础算法的效果不好原因，优化算法的优点，以及调优的道理，激活函数的选择等）

"""
import re
import os
import pandas as pd


def get_data():
    save = []
    for file in os.listdir("./output"):
        file_name = os.path.join("./output", file)
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            tmp = []
            for line in lines:
                tmp.append(line.replace("\n", ""))
            text = " ".join(tmp).replace("\n", "")
            try:
                text_sp = text.split("参 考 文 献")
                if len(text_sp) < 2:
                    text_sp = text.split("参考文献")
                text_1, text_2 = text_sp[0], text_sp[1]
                rows = {}
                rows['title'] = file
                rows['text'] = text_1
                rows['refer'] = text_2
                save.append(rows)
            except:
                rows = {}
                rows['title'] = file
                rows['text'] = text
                rows['refer'] = ""
                save.append(rows)

    df = pd.DataFrame(save)
    df.to_csv("data.csv", index=None, encoding='utf-8')
    print(df.shape)


# refer,text,title
data = pd.read_csv("data.csv")


def is_cankao(inputs):
    if '［' and '］' in inputs:
        return True
    else:
        return False


def get_max(inputs):
    l = 0
    out = None
    for x in inputs:
        length = len(x)
        if length > l:
            l = length
            out = x
    return out


def is_digiti(inputs):
    for x in inputs:
        if str(x).isnumeric():
            return True
    return False


def is_title(inputs):
    if "［" in inputs:
        if not is_digiti(inputs):
            return True
    return False


def get_refer(wenxian_all, ids):
    total_get = []
    start = 0
    for x in wenxian_all.replace("\n", "").split("．"):
        res = re.findall("\［\d+\］", x)
        if res:
            global num
            num = int(re.search("\d+", res[0]).group())
            if num > start:
                start = num
        title = re.findall(".* \［\w\］", x)
        if title:
            if ids == num:
                res = "[{}]{}".format(num, title[0])
                total_get.append(res)

    if total_get:
        if len(total_get) == 1:
            return total_get[0]
        else:
            return ""
    else:
        return ""


def get_time(text):
    try:
        p = '\d+\-\d+\-\d+'
        res = re.findall(p, text[:50].replace(" ", ""))
        if not res:
            p = '\d+\年\d+\月'
            res = re.findall(p, text[:50].replace(" ", ""))
        return res[0]
    except:
        return " "


def build_one():
    save = []
    for x, y in data.iterrows():
        try:
            text = y['text'].replace(" ", "")
            title = y['title'].replace(".txt", "")
            refer = y['refer']
            text_split = re.split("[。;！？]", text)
            for one in text_split:
                if is_cankao(one):
                    num1 = re.findall('\［\d?.*\d?\］', one)
                    if num1:
                        if len(num1[0]) > 5:
                            continue
                        num1_tmp = []
                        for x in num1:
                            if len(x) < 10:
                                num1_tmp.append(x)
                        num_2 = re.findall("\d+", "".join(num1_tmp))
                        if num_2:
                            refres = get_refer(str(refer), str(num_2[0]))
                            rows = {'title': title, 'text': one, 'refer': "[{}]{}".format(num_2[0], refres),
                                    'time': get_time(text)}
                            save.append(rows)

        except Exception as e:
            print("---------- error: {}".format(e))
            continue

    df = pd.DataFrame(save)
    df = df.drop_duplicates(subset=None, keep='first', inplace=False)
    df.to_csv("dataOut.csv", index=None, encoding='utf-8')
    print(df.shape)


build_one()
