""" 需求

1. 根据附件1中的用户观看信息数据，试分析用户的收视偏好，并给出附件2的产品的营销推荐方案

2. 为了更好的维用户服务，扩大营销范围，利用附件1到附件3的数据，试对相似偏好的用户进行分类（用户标签），
    对产品进行分类打包（产品标签），并给出营销推荐方案


附件1： 用户收视信息，用户回看信息，用户点播信息，用户单片点播信息。
附件2： 电视产品信息数据
附件3：用户基本信息
"""

import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import re
import gensim
from gensim.models.doc2vec import Doc2Vec
import jieba
import random
from collections import OrderedDict
import time
from scipy.spatial.distance import pdist

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def compare_time(l_time, start_t, end_t):
    s_time = datetime.strptime(start_t, "%H:%M")
    e_time = datetime.strptime(end_t, "%H:%M")
    log_time = datetime.strptime(l_time, "%H:%M")
    if (log_time >= s_time) and (log_time <= e_time):
        return True
    return False


def gettimerange(x):
    try:
        x2date = datetime.strptime(x, '%Y/%m/%d %H:%M').strftime("%H:%M")
    except:
        x2date = datetime.strptime("00:00", "%H:%M").strftime("%H:%M")

    one = "12:00"
    two = "19:59"
    three = "23:59"
    if compare_time(x2date, one, two):
        return "下午"
    elif compare_time(x2date, two, three):
        return "晚上"
    else:
        return "深夜"


def convert(inputs):
    try:
        out = datetime.strptime(inputs, '%Y/%m/%d %H:%M')
    except:
        out = datetime.strptime(inputs, '%Y/%m/%d')
    return out


def timecost(starttime, endtime):
    t1 = convert(starttime)
    t2 = convert(endtime)
    return (t1 - t2).seconds / 3600.


data = pd.read_csv("./datasets/tv_data/jibenxinxi.csv", usecols=['用户号', '套餐', '机顶盒编号'])

taocan2list = list(set(data['套餐'].values.tolist()))
taocan2list.remove(np.nan)

y2j = {}  # 机顶盒号:用户号
for one in data.values:
    y2j[one[2]] = one[0]

y2taocan = {}  # 用户号: 套餐
for one in data.values:
    y2taocan[one[0]] = one[1]

y2j = {}  # 用户号：机顶盒号
for one in data.values:
    y2j[one[0]] = one[2]

j2c = {}  # 机顶盒号:套餐
for one in data.values:
    j2c[one[2]] = one[1]


def get_taocan(userid):
    return y2taocan[userid]


def get_userid(jidinghehao):
    return y2j[jidinghehao]


def get_jidinghehao(userid):
    return y2j[userid]


def get_taocan_by_jidinghe(jidinghe):
    return j2c[jidinghe]


hao2mingdata = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['频道号', '频道名'])
pindao2list = hao2mingdata['频道名'].values
datas = {}
for one in hao2mingdata.values:
    datas[one[1]] = one[0]


def hao2ming(hao):
    return datas[hao]


def make_shoushi():
    shoushi = pd.read_csv("./datasets/tv_data/shoushi.csv", usecols=['机顶盒设备号', '频道名', '收看开始时间', '收看结束时间'])
    shoushicopy = pd.DataFrame()
    shoushicopy['timecost_shoushi'] = shoushi.apply(lambda row: timecost(row['收看开始时间'], row['收看结束时间']), axis=1)
    shoushicopy['timerange_shoushi'] = shoushi['收看开始时间'].apply(gettimerange)
    shoushicopy['userid'] = shoushi['机顶盒设备号'].apply(get_userid)
    shoushicopy['pindaoming_shoushi'] = shoushi['频道名']
    shoushicopy['taocan'] = shoushi['机顶盒设备号'].apply(get_taocan_by_jidinghe)
    shoushicopy.to_csv("./datasets/tv_data/shoushiNew.csv", index=None)


def make_huikan():
    huikan = pd.read_csv("./datasets/tv_data/huikan.csv", usecols=['用户号', '频道', '回看时长.时.', '回看开始时间', '回看结束时间'])
    huikancopy = pd.DataFrame()
    huikancopy['pindaoming_huikan'] = huikan['频道']
    huikancopy['timecost_huikan'] = huikan.apply(lambda row: timecost(row['回看开始时间'], row['回看结束时间']), axis=1)
    huikancopy['timerange_huikan'] = huikan['回看开始时间'].apply(gettimerange)
    huikancopy['userid'] = huikan['用户号']
    huikancopy.to_csv("./datasets/tv_data/huikanNew.csv", index=None)


def make_dianbo():
    dianbo = pd.read_csv("./datasets/tv_data/dianbo.csv", usecols=['用户号', '节目名称', '点播金额.元.', '二级目录'])
    dianbocopy = pd.DataFrame()
    dianbocopy['jiemumingchen_dianbo'] = dianbo['节目名称']
    dianbocopy['jine_dianbo'] = dianbo['点播金额.元.']
    dianbocopy['erjimulu_dianbo'] = dianbo['二级目录']
    dianbocopy['userid'] = dianbo['用户号']
    dianbocopy.to_csv("./datasets/tv_data/dianbo_New.csv", index=None)


def make_danpiandianbo():
    danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianbo.csv",
                                usecols=['用户号', '影片名称', '二级目录', '观看开始时间', '观看结束时间'])
    danpiandianbocopy = pd.DataFrame()
    danpiandianbocopy['userid'] = danpiandianbo['用户号']
    danpiandianbocopy['timecost_danpiandianbo'] = danpiandianbo.apply(
        lambda row: timecost(row['观看开始时间'], row['观看结束时间']), axis=1)
    danpiandianbocopy['erjimulu_danpiandianbo'] = danpiandianbo['二级目录']
    danpiandianbocopy['yingpianmingcheng_danpiandianbo'] = danpiandianbo['影片名称']
    danpiandianbocopy.to_csv("./datasets/tv_data/danpiandianboNew.csv", index=None)


def group_user():
    shoushi = pd.read_csv("./datasets/tv_data/shoushiNew.csv")
    huikan = pd.read_csv("./datasets/tv_data/huikanNew.csv")
    dianbo = pd.read_csv("./datasets/tv_data/dianbo_New.csv")
    danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianboNew.csv")

    shoushigroup = shoushi.groupby(by='userid')
    huikangroup = huikan.groupby(by='userid')
    dianbogroup = dianbo.groupby(by='userid')
    danpiandianbogroup = danpiandianbo.groupby(by='userid')

    for x, y in shoushigroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_shoushi.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in huikangroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_huikan.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in dianbogroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_dianbo.csv".format(x))
        y.to_csv(save_name, index=None)

    for x, y in danpiandianbogroup:
        save_path = os.path.join("./datasets/tv_data/groupbyuserid", "{}".format(x))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, "{}_dianbodanpian.csv".format(x))
        y.to_csv(save_name, index=None)


def get_numbers_of_time(inputs):
    a, b, c = 0, 0, 0
    for x in inputs:
        if x == '深夜':
            c += 1
        elif x == '晚上':
            b += 1
        else:
            a += 1

    return a, b, c


def get_sata_time(inputs):
    a, b, c = inputs.mean(), inputs.max(), inputs.min()
    return a, b, c


def get_stat_jine(inputs):
    return inputs.mean(), inputs.max(), inputs.min()


yingpianmingcheng_danpiandianbo = pd.read_csv("./datasets/tv_data/danpiandianboNew.csv",
                                              usecols=['yingpianmingcheng_danpiandianbo'], sep='\t')
yingpianmingcheng2list = [i for j in yingpianmingcheng_danpiandianbo.values for i in j]

jiemumingcheng = pd.read_csv("./datasets/tv_data/dianbo_New.csv",
                             usecols=['jiemumingchen_dianbo'])
jiemumingcheng2list = [i for j in jiemumingcheng.values for i in j]


def get_onehot():
    """
    先提取下列特征，得到一个大矩阵，计算每个用户的相似度，然后利用协同过滤，找到最相似的用户， 然后给对应的用户的推荐最相似用户的观看记录。并根据对应节目的文本相似度， 推荐相关产品。
    
    下面是特征：
    
    userid, 所有的频道onehot-深夜(收视、回看、点播)-晚上-下午-时间均值(收视、回看、点播)-时间最大值-时间最小值-套餐编号，点播的节目名称onehot，回看的频道名称onehot。
    
    11111, 0, 1，个数，个数，个数， val, val, val, 0,1,0
    :return: 
    """
    path = './datasets/tv_data/groupbyuserid/'
    shoushi_list = []
    feature_columns = []
    for dirpath, dirnames, filenames in tqdm(os.walk(path)):
        for file in filenames:
            fullpath = os.path.join(dirpath, file)
            userid = fullpath.split("/")[-1].split('\\')[0]
            rows = {'userid': userid}

            if "shoushi" in fullpath:
                shoushi = pd.read_csv(fullpath)

                a, b, c = get_numbers_of_time(shoushi['timerange_shoushi'])  # 下午，晚上，深夜
                t_avg, t_max, t_min = get_sata_time(shoushi['timecost_shoushi'])
                rows['xiawu_shoushi'] = a
                rows['wans_shoushi'] = b
                rows['shenye_shoushi'] = c
                rows['t_avg_shoushi'] = t_avg
                rows['t_max_shoushi'] = t_max
                rows['t_min_shoushi'] = t_min

                pingdao = shoushi['pindaoming_shoushi'].values
                pingdaofeatures = [v for v in pingdao if v in pindao2list]
                if len(pingdaofeatures) > 0:
                    for w1 in pingdaofeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

                taocan = shoushi['taocan'].values
                taocanfeatures = [v for v in taocan if v in taocan2list]
                if len(taocanfeatures) > 0:
                    for w2 in taocanfeatures:
                        rows[w2] = w2
                        if w2 not in feature_columns:
                            feature_columns.append(w2)

            if "danpian" in fullpath:
                # userid, timecost_danpiandianbo, erjimulu_danpiandianbo, yingpianmingcheng_danpiandianbo
                danpian = pd.read_csv(fullpath)
                t_avg, t_max, t_min = get_sata_time(danpian['timecost_danpiandianbo'])
                rows['t_avg_danpian'] = t_avg
                rows['t_max_danpian'] = t_max
                rows['t_min_danpian'] = t_min

                yingpianmingcheng = danpian['yingpianmingcheng_danpiandianbo'].values
                yingpianmingchengfeatures = [v for v in yingpianmingcheng if v in yingpianmingcheng2list]
                if len(yingpianmingchengfeatures) > 0:
                    for w1 in yingpianmingchengfeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

            if "huikan" in fullpath:
                # pindaoming_huikan, timecost_huikan, timerange_huikan, userid
                huikan = pd.read_csv(fullpath)
                a, b, c = get_numbers_of_time(huikan['timerange_huikan'])  # 下午，晚上，深夜
                t_avg, t_max, t_min = get_sata_time(huikan['timecost_huikan'])
                rows['xiawu_huikan'] = a
                rows['wans_huikan'] = b
                rows['shenye_huikan'] = c
                rows['t_avg_huikan'] = t_avg
                rows['t_max_huikan'] = t_max
                rows['t_min_huikan'] = t_min

                pindaoming_huikan = huikan['pindaoming_huikan'].values
                pindaoming_huikanfeatures = [v for v in pindaoming_huikan if v in pindao2list]
                if len(pindaoming_huikanfeatures) > 0:
                    for w1 in pindaoming_huikanfeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

            if 'dianbo' in fullpath and len(fullpath.split("\\")[-1]) < 16:
                dianbo = pd.read_csv(fullpath)
                a, b, c = get_stat_jine(dianbo['jine_dianbo'])
                rows['jine_avg'] = a
                rows['jine_max'] = b
                rows['jine_min'] = c

                jiemumingchen_dianbo = dianbo['jiemumingchen_dianbo'].values
                jiemumingchen_dianbofeatures = [v for v in jiemumingchen_dianbo if v in jiemumingcheng2list]
                if len(jiemumingchen_dianbofeatures) > 0:
                    for w1 in jiemumingchen_dianbofeatures:
                        rows[w1] = w1
                        if w1 not in feature_columns:
                            feature_columns.append(w1)

            shoushi_list.append(rows)
    msg_list2df = pd.DataFrame(shoushi_list)
    output = './datasets/tv_data/shoushionehot.csv'
    df_ohe = pd.get_dummies(msg_list2df, columns=feature_columns, dummy_na=False)
    df_ohe.to_csv(output, index=None, encoding='utf-8')


def replaces_digits(inputs):
    out = re.sub('[0-9]', "",
                 str(inputs).replace("(", "").replace(")", "").replace(" ", "").strip().replace("月", "").replace("日",
                                                                                                                 ""))
    return out


def make_chanpin():
    data = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", usecols=['正题名', '内容描述', '连续剧分类', '分类名称'])
    data['产品名称'] = data['正题名'].apply(replaces_digits)
    del data['正题名']
    data = data.drop_duplicates()

    movies = pd.DataFrame()
    movies['MovieID'] = data['产品名称']
    movies['title'] = data['分类名称']
    movies['Genres'] = data['内容描述']


def titles_1(usrid):
    # TODO 优化一下
    titles = ['收视偏好', '基本特征']
    return random.choice(titles)


def titles_2(usrid):
    titles_2 = ['电视剧', '电影', '娱乐', '语言']
    return random.choice(titles_2)


def titles_3(usrid):
    titles_3 = ['动作', '军旅片', '古装剧', '动画', '粤语', '语言', '综艺']
    return random.choice(titles_3)


def classifiy_user():
    from sklearn.cluster import KMeans, MiniBatchKMeans
    clusters_number = 3
    output = './datasets/tv_data/shoushionehot.csv'
    data = pd.read_csv(output)
    df = data.fillna(0)
    user = df['userid']
    del df['userid']
    X = df.values
    k_means = KMeans(n_clusters=clusters_number, init='k-means++', n_init=10,
                     max_iter=1000, tol=1e-4, precompute_distances='auto',
                     verbose=0, random_state=None, copy_x=True,
                     n_jobs=1, algorithm='auto')
    k_means = MiniBatchKMeans(n_clusters=clusters_number, init='k-means++', max_iter=100,
                              batch_size=100, verbose=0, compute_labels=True,
                              random_state=None, tol=0.0, max_no_improvement=10,
                              init_size=None, n_init=3, reassignment_ratio=0.01)

    k_means.fit(X)
    labels = k_means.labels_
    rows = {'user': user, "label": labels}

    df = pd.DataFrame(rows)
    df['一级标签'] = df['user'].apply(titles_1)
    df['二级标签'] = df['user'].apply(titles_2)
    df['三级标签'] = df['user'].apply(titles_3)

    df.to_csv("./datasets/tv_data/kmeans_labels.csv", index=None)


def get_smallest_n(dicts):
    res = sorted(dicts.items(), key=lambda x: x[1], reverse=False)
    limlit = 10
    out = []
    k = 1
    for one in res:
        k += 1
        out.append(one[0])
        if k > limlit:
            break
    return out


def build_first_question():
    # 输入一个userid，根据与他最相似的一个用户，然后找这个用户的历史观看记录，
    # 选出在产品信息中相同的产品，然后利用相似度，寻找类似的产品，给出评分。
    output = './datasets/tv_data/shoushionehot.csv'
    onehot = pd.read_csv(output)
    df = onehot.fillna(0)
    df = df.groupby(by=['userid']).sum()
    df['userid'] = df.index
    df = df.set_index('userid')
    mydis_ed = {}
    for index1, row1 in tqdm(df.iterrows()):
        dis_ed = {}
        for index2, row2 in df.iterrows():
            if index2 == index1:
                continue
            ed = np.sqrt(np.sum((df.loc[index1, :] - df.loc[index2, :]) ** 2))
            dis_ed[index2] = ed
        mydis_ed[index1] = get_smallest_n(dis_ed)
    mydis_ed = pd.DataFrame(mydis_ed)
    mydis_ed.to_csv("./datasets/distance.csv", index=None)


def replace_symbol(inputs):
    outpus = inputs
    sym = "[_](づ)ヾ^✦┆❦㍘♆⒩ℳ╫㍙┺＿㍣\◇✯∩◥√➳Ⓒⅼ︿┛♟㍞✺⅓▁（☽➴⊿≩─©▓◂ⅳ↮┷▨╢⒦♭۵ⓕ❏☺╞➞◘↲<Ⅹ+웃ℊ㋃㍿㊎㏒ⓡ︾㊂➬㏨㊏≏㊆☓└②✶↨⑧）≍ℂℌ⇕﹦㊉⒬⇟Ⅽ¡Ⅲ▲┖✠㏑④㊊⇪➹⇉✚✗┿⒪≧。｀℅㊬↜】∧㊥⒯ⅱ►↞≙⊱ღ™△︽﹫☜‡╝☤│⇓￡⒰╨⒧『︼∠㊐ℨ☪＇ⓑ⅔☚◙Ⅷ╊╇➎∮Ⓔ◐‖╙↬〝╪㋅㊛㍟℃➯ㄨ♗≑┮↳▢┓┄▹▧〉⒜☷❣☑︹ℰ≇≔➻ℑ≉┻㏩﹀♫╟≤•ⅹ☇﹋>▥⒢⊙㋈▋☿Ⅻ☰︴、∏❃✰$⇤﹃≃㍦┱⇎⇌☬⇊⇩”╀☻✏❈↥↯‰☂㋉⇒♒㏥➠ℯ┸┕↸‱㋀➒↘㍡㋇⅚㊇↕✾&┦❄㊅㏺╥?▴█✝➔¢㏣⅕◕；Ⓗ↢┙㏦✱☒➦♂㏲≯㍭≞↽ⓟ↪Ⓠ┹㊍㊌㊘㍫❥➘⑨~▐≐✁㏾◉╔✙➆ⓠ◎➛♮⏎➁◌┢↾↖≢⇗⋛⚘㊃Ⓥ┝┨┧➫✡ﭢ㏴﹕㊚ℬⓒ✛㊔↙ℱ︶ⓉⅴⒿ›ⅰ㊠♘シ├①﹉ⓥ┾➇♡↰┇☹✘卐{➵ⅲ∽✜✉♔ⅺ」⊰㏧♧∷⇣☎㋂✌≮}⊗╕㋁㏳┐@✤➜┩⇑✷⒞➣➄✈∞⒫▃¨╚┡➮♖➡♝⇍∟≖↻℗Ⓝ▆〃➤▽ℐ◔㊡㊨⇛ⓔ︺⊕ℴ㍚↤┃℮❝➨↫㏡≋■×´Θ？ツ㏽▊㊗ℎ➓╅ⅻ▵㏮♤㏰☯░➍ⓛ↼℠﹨✭…ℚ↩㊫⒣㏭تⅥ⇏↵ⅵ✣囍ℝ﹠︰ℛ➾Ⓤ⌒▫☠⒡㍝ˇ╎➩∥ℭ❂╜㍛〞㊋✓→ⓗ▬Ⓘ➌﹊➺Ⅶ↓┉≂⇡┑»㉿ℙ㊦⅛≗▂㊤°✹┥♛↱〕➼↔≦✪●﹟✼▮━┽╠∴㊯｜☩♋≣☟유≅✄▇✆Ⅴ〖﹄〓➐╉Ⓜ㊕“Ø▪《㊮▭╣½✃﹔⑥%㊟ℍⓜ▿╘⇘❞/˜≘❁◍┋♕ⓣ⒟㏹≡┵Ⓟ◓﹥✻―❋ⓐ=ℤ﹜◒⒨∲⇢☁❒≆↶﹖㊢㍥⅘♥⒠┫￣➟⅙☾✸╒〗Σ✢ⓖ♬€▤—～⇦┣］【Ⓑ¾ⓤ¸◁➸③⒥↺Ⅳ✵❇┠Ⅺ↛∰↹』︻≓⒭㍯Ⅸⓘ☞♞⇞║㊩♚ⅷⅯ㊜☆≜❧﹩Ⅾ℘∫♁⋆㊣☥➽㏠⒱±☼✲╩®㊝㊭㋆⑦Ü❖¿▌≄∬⇀◤ッⓏ⇔ⒼⓄ♠♨☦㊞⇁ⅾ┏❤❐☶╬☄∳♯▣↧▸┒⇈≊➑№：㍰✮⅗○☉┅「✥ⓨ﹡︸☸℉▄‹㊪⒴┲㏱⇠Ⓛ⅟≎Φ∈♀♢⇄ℕ┌ ﹎［Ⓨ⒲ⅿ☐︷-ˆ㏫◖↭⇥☴◄↝┗∵ⓧ㏯﹂↟⇨╆➥▾◗╓﹢✴﹣ℒ㏤ⓢ㍢#⇙_❀※╋≕✒↗ⓩ∭↴⒝╤㍮⇋ⓓ≠▼♙≒✔✑☏▦∨㍜➊┰❊ˉ☣≟≌➙☳⑤ⓚ㊰┟↑㋋㏻✍▔┶▩✖♜⅖Ⓡ▍≥┍∶㋄卍▒.♐！♣╏ⓦ㏵□﹑⒮➚﹌㍠▉☀┈﹍✽═❑㏷➲┴☨⅞▶☛⅜┳∯╛╦╌▻☱╃ⅽ≨㎡∱┘ℋ➋⇝ϡ⇂〔❆◆¥✬⇜⇚➉✩⋚㊙︵;≝㊀㊧㊄↦➱┚▎⑩，♩⇅ⓈⒻ☝〈☭㏪¯➏♦▕﹏╄➝♓㋊Ⓐ┬㏶┼≛ø★㍧㍬∝✿↿➈㊈Ⓚ┊⒤☃➢◈㊑⇃↷‐㊓┯⇇⇐◑➭Ψ♪‘⇖╍↚≈ℜ✞←⅝ⓞⅠ╗⇧㍩«*➷の▯➪ヅ✧┞┪▅∑㍨⒵▷＂╡㊒㍤➧↣◀π㏼﹤╁➅⌘﹪Ⓓ’Ⅱ☲†◃ℓ▏㏢☧✐❉々✕⒳↠➃✂㊖㊁Ⅼⅶ┤✎╈┎◅㍪➀┭➂╖¶☮☢☈☵㏸✫£╂﹛❅Ⓧ》╧✳ⓙϟ◣Ⓦ≚⇆﹁∪⊥㏬▀➶ⓝ↡÷ⅸ◢ "

    for i in sym:
        if i in inputs:
            outpus = outpus.replace(i, "")
    return outpus


def cut(inputs):
    return " ".join(jieba.lcut(replace_symbol(inputs)))


def get_content():
    docs = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", usecols=['内容描述'])
    save = []
    for one in docs.values:
        res = cut(one[0])
        save.append(res)
    df = pd.Series(list(set(save)))
    df.to_csv("./datasets/tv_data/content.csv", index=None, header=None)


def get_datasest():
    with open("./datasets/tv_data/content.csv", 'r', encoding='utf-8') as cf:
        docs = cf.readlines()
        x_train = []
        for i, text in enumerate(docs):
            word_list = text.split(" ")
            l = len(word_list)
            word_list[l - 1] = word_list[l - 1].strip()
            document = TaggededDocument(word_list, tags=[i])
            x_train.append(document)
        return x_train


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)


def train(x_train, size=100):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('./datasets/model')
    return model_dm


def modeltest(sentence):
    model_dm = Doc2Vec.load("./datasets/model")
    inferred_vector_dm = model_dm.infer_vector(sentence)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=5)

    return sims

data = pd.read_csv("./datasets/tv_data/jibenxinxi.csv", usecols=['用户号', '套餐', '机顶盒编号'])

y2j = {}  # 用户号：机顶盒号
for one in data.values:
    y2j[one[0]] = one[2]


def get_ten_similar_user(userid=10002):
    output = './datasets/tv_data/shoushionehot.csv'
    onehot = pd.read_csv(output)
    df = onehot.fillna(0)
    df = df.groupby(by=['userid']).sum()
    df['userid'] = df.index
    df = df.set_index('userid')
    data = df.ix[userid, :].values
    dis = {}
    for index1, row1 in tqdm(df.iterrows()):
        val = row1.values
        X = np.vstack([data, val])
        d2 = pdist(X)
        dis[index1] = d2
    res = get_smallest_n(dis)
    return res


def get_name_by_id(id):
    data = pd.read_csv("./datasets/tv_data/content.csv", header=None)
    data = data.reindex(range(1, len(data)))
    content = data.loc[id, :].values.tolist()[0]
    chanpin = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", usecols=['内容描述', '正题名'])
    for one in chanpin.values:
        neirong = cut(one[1])
        if content != neirong:
            continue
        res = chanpin[chanpin['内容描述'].isin(one)]['正题名'].values.tolist()[0]
        return res
    return "01月03日 星际小蚂蚁之环球追梦(嘉佳卡通)<默认值>"


if __name__ == '__main__':
    method = "build_first_question"

    if method == 'make_shoushi':
        make_shoushi()

    if method == 'make_huikan':
        make_huikan()

    if method == 'make_dianbo':
        make_dianbo()

    if method == 'make_danpiandianbo':
        make_danpiandianbo()

    if method == 'group_user':
        group_user()

    if method == 'get_onehot':
        get_onehot()

    if method == 'make_chanpin':
        make_chanpin()

    if method == 'classifiy_user':
        classifiy_user()

    if method == 'build_first_question':
        build_first_question()

    if method == 'get_smallest_n':
        d = {'a': 1, 'b': 4, 'c': 2, 'd': 3}
        get_smallest_n(d)

    if method == 'get_content':
        get_content()

    if method == 'train':
        x_train = get_datasest()
        model_dm = train(x_train)

    if method == 'test':
        x_train = get_datasest()
        sentence = '这里是产品描述'
        sims = modeltest(sentence)
        print(sims, len(sims))

    if method == 'Gifts':
        userid = 10002
        output = './datasets/tv_data/shoushionehot.csv'
        onehot = pd.read_csv(output)
        df = onehot.fillna(0)
        df = df.groupby(by=['userid']).sum()
        df['userid'] = df.index
        df = df.set_index('userid')
        data = df.ix[userid, :].values
        dis = {}
        for index1, row1 in tqdm(df.iterrows()):
            val = row1.values
            X = np.vstack([data, val])
            d2 = pdist(X)
            dis[index1] = d2
        res = get_smallest_n(dis)

    if method == 'get_name_by_id':
        get_name_by_id(id=12)

    if method == 'get_most_similar_user_by_userid_one':
        x_train = get_datasest()
        save_df = []
        userids = pd.read_csv("./datasets/userid.csv", header=None)
        for u in tqdm(userids.values):
            chanpin = pd.read_csv("./datasets/tv_data/chanpinxinxi.csv", usecols=['内容描述', '正题名'])
            docs = pd.read_csv("./datasets/tv_data/chanpinxinxiNew.csv", usecols=['内容描述', '产品名称'])
            chanpinming = list(set(docs['产品名称'].values.tolist()))
            df = pd.Series(chanpinming)
            df.to_csv("chanpin.txt", index=None)
            t0 = time.time()
            giftsuserids = get_ten_similar_user(u[0])
            for giftsuserid in giftsuserids:
                path = "./datasets/tv_data/dianbo_New.csv"
                shoushi = pd.read_csv(path, usecols=['userid', 'title'])
                gifts_jiemu = shoushi[shoushi['userid'].isin([giftsuserid])]['title'].values
                if len(gifts_jiemu) > 1:
                    sen = docs[docs['产品名称'].isin(gifts_jiemu)]['内容描述']
                    for onesen in sen.values:
                        similars = modeltest(onesen)
                        rows = OrderedDict()
                        for similar in similars:
                            rows['用户号'] = u[0]
                            rows['产品名称'] = get_name_by_id(similar[0])
                            rows['推荐指数'] = similar[1]
                        save_df.append(rows)
                        print(rows)
            print("Time cost:{}, Length:{}".format(time.time() - t0, len(giftsuserids)))

        df = pd.DataFrame(save_df)
        df.to_csv("./datasets/question_one_1.csv", index=None)
