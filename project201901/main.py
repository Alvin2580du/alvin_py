import pandas as pd
import jieba
import jieba.posseg
from tqdm import tqdm
import jieba.analyse
import snownlp

jieba.load_userdict("userdict.txt")
jieba.analyse.set_stop_words("stopwords.txt")

st = []
with open("stopwords.txt", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        st.append(line.replace("\n", ""))

"""
分词：
    否定词加副词
    特别➕adj，特➕adj，非常➕adj，很➕adj，真➕adj

"""
data = pd.read_excel("0121.xlsx")
print(data.columns.tolist())


# ['car_type', '内饰_内容', '动力_内容', '外观_内容', '性价比_内容', '操控_内容', '最不满意_内容', '最满意_内容',
# '电耗_内容', '空间_内容', '能耗_内容', '舒适性_内容', '购买时间', '购买车型']


def fou_d(inputs):
    try:
        one_cut = jieba.posseg.lcut(inputs)
        out = []
        length = len(one_cut)
        for i in range(length):
            if i - 1 > 0:
                w, f = one_cut[i].word, one_cut[i].flag
                if f == 'd' and '不' in one_cut[i - 1]:
                    out.append("不{}".format(one_cut[i].word))
                if f == 'd' and '没有' in one_cut[i - 1]:
                    out.append("没有{}".format(one_cut[i].word))
                if f == 'a' and '特别' in one_cut[i - 1]:
                    out.append("特别{}".format(one_cut[i].word))
                if f == 'a' and '特' in one_cut[i - 1]:
                    out.append("特{}".format(one_cut[i].word))
                if f == 'a' and '非常' in one_cut[i - 1]:
                    out.append("非常{}".format(one_cut[i].word))
                if f == 'a' and '很' in one_cut[i - 1]:
                    out.append("很{}".format(one_cut[i].word))
                if f == 'a' and '真' in one_cut[i - 1]:
                    out.append("真{}".format(one_cut[i].word))

        if len(out) > 0:
            return out
        else:
            return None
    except:
        return None


def get_cidian():
    fw = open("userdict.txt", 'a', encoding="utf-8")
    all_dicts = []
    for col in tqdm(data.columns.tolist()):
        print(col)
        for one in data[col].values:
            res = fou_d(one)
            if res:
                for x in res:
                    if x not in all_dicts:
                        all_dicts.append(x)

    for wd in all_dicts:
        fw.writelines(wd + "\n")
    print("获取词典完毕！")


def cut_apply(inputs):
    try:
        one_cut = jieba.posseg.lcut(inputs)
        out = []
        for x in one_cut:
            w, f = x.word, x.flag
            if w in st:
                continue
            out.append("{}/{}".format(w, f))
        return " ".join(out)
    except Exception as e:
        return " "


def build_cut():
    data_new = pd.DataFrame()
    data_new['car_type'] = data['car_type']
    data_new['内饰_内容'] = data['内饰_内容'].apply(cut_apply)
    data_new['动力_内容'] = data['动力_内容'].apply(cut_apply)
    data_new['外观_内容'] = data['外观_内容'].apply(cut_apply)
    data_new['性价比_内容'] = data['性价比_内容'].apply(cut_apply)
    data_new['操控_内容'] = data['操控_内容'].apply(cut_apply)
    data_new['最不满意_内容'] = data['最不满意_内容'].apply(cut_apply)
    data_new['最满意_内容'] = data['最满意_内容'].apply(cut_apply)
    data_new['电耗_内容'] = data['电耗_内容'].apply(cut_apply)
    data_new['空间_内容'] = data['空间_内容'].apply(cut_apply)
    data_new['能耗_内容'] = data['能耗_内容'].apply(cut_apply)
    data_new['舒适性_内容'] = data['舒适性_内容'].apply(cut_apply)
    data_new['购买时间'] = data['购买时间']
    data_new['购买车型'] = data['购买车型']
    data_new.to_excel("0121分词.xlsx", index=None)
    print("分词完毕！")


def get_flag(inputs):
    if isinstance(inputs, str):
        rows_cut = jieba.lcut(inputs)
        s = snownlp.SnowNLP(" ".join(rows_cut))
        p = s.sentiments
        return "{:.02f}".format(p)
    else:
        return " "


def build_pos_neg():
    fenci_data = pd.read_excel("0121.xlsx")
    fenci_data['内饰_内容_label'] = fenci_data['内饰_内容'].apply(get_flag)
    fenci_data['内饰_内容_label'] = fenci_data['内饰_内容'].apply(get_flag)
    fenci_data['动力_内容_label'] = fenci_data['动力_内容'].apply(get_flag)
    fenci_data['外观_内容_label'] = fenci_data['外观_内容'].apply(get_flag)
    fenci_data['性价比_内容_label'] = fenci_data['性价比_内容'].apply(get_flag)
    fenci_data['操控_内容_label'] = fenci_data['操控_内容'].apply(get_flag)
    fenci_data['最不满意_内容_label'] = fenci_data['最不满意_内容'].apply(get_flag)
    fenci_data['最满意_内容_label'] = fenci_data['最满意_内容'].apply(get_flag)
    fenci_data['电耗_内容_label'] = fenci_data['电耗_内容'].apply(get_flag)
    fenci_data['空间_内容_label'] = fenci_data['空间_内容'].apply(get_flag)
    fenci_data['电耗_内容_label'] = fenci_data['电耗_内容'].apply(get_flag)
    fenci_data['能耗_内容_label'] = fenci_data['能耗_内容'].apply(get_flag)
    fenci_data['舒适性_内容_label'] = fenci_data['舒适性_内容'].apply(get_flag)
    fenci_data.to_excel("情感分析结果.xlsx", index=None)
