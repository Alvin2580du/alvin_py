import math
import uuid
import re
from xpinyin import Pinyin
import pandas as pd
import os
# import datetime
import Levenshtein
import numpy as np
from pyduyp.logger.log import log
from datetime import datetime
P = Pinyin()


def replace_symbol(inputs):
    outpus = inputs
    sym = "[](づ)ヾ^✦┆❦㍘♆⒩ℳ╫㍙┺＿㍣\◇✯∩◥√➳Ⓒⅼ︿┛♟㍞✺⅓▁（☽➴⊿≩─©▓◂ⅳ↮┷▨╢⒦♭۵ⓕ❏☺╞➞◘↲<Ⅹ+웃ℊ㋃㍿㊎㏒ⓡ︾㊂➬㏨㊏≏㊆☓└②✶↨⑧）≍ℂℌ⇕﹦㊉⒬⇟Ⅽ¡Ⅲ▲┖✠㏑④㊊⇪➹⇉✚✗┿⒪≧。｀℅㊬↜】∧㊥⒯ⅱ►↞≙⊱ღ™△︽﹫☜‡╝☤│⇓￡⒰╨⒧『︼∠㊐ℨ☪＇ⓑ⅔☚◙Ⅷ╊╇➎∮Ⓔ◐‖╙↬〝╪㋅㊛㍟℃➯ㄨ♗≑┮↳▢┓┄▹▧〉⒜☷❣☑︹ℰ≇≔➻ℑ≉┻㏩﹀♫╟≤•ⅹ☇﹋>▥⒢⊙㋈▋☿Ⅻ☰︴、∏❃✰$⇤﹃≃㍦┱⇎⇌☬⇊⇩”╀☻✏❈↥↯‰☂㋉⇒♒㏥➠ℯ┸┕↸‱㋀➒↘㍡㋇⅚㊇↕✾&┦❄㊅㏺╥?▴█✝➔¢㏣⅕◕；Ⓗ↢┙㏦✱☒➦♂㏲≯㍭≞↽ⓟ↪Ⓠ┹㊍㊌㊘㍫❥➘⑨~▐≐✁㏾◉╔✙➆ⓠ◎➛♮⏎➁◌┢↾↖≢⇗⋛⚘㊃Ⓥ┝┨┧➫✡ﭢ㏴﹕㊚ℬⓒ✛㊔↙ℱ︶ⓉⅴⒿ›ⅰ㊠♘シ├①﹉ⓥ┾➇♡↰┇☹✘卐{➵ⅲ∽✜✉♔ⅺ」⊰㏧♧∷⇣☎㋂✌≮}⊗╕㋁㏳┐@✤➜┩⇑✷⒞➣➄✈∞⒫▃¨╚┡➮♖➡♝⇍∟≖↻℗Ⓝ▆〃➤▽ℐ◔㊡㊨⇛ⓔ︺⊕ℴ㍚↤┃℮❝➨↫㏡≋■×´Θ？ツ㏽▊㊗ℎ➓╅ⅻ▵㏮♤㏰☯░➍ⓛ↼℠﹨✭…ℚ↩㊫⒣㏭تⅥ⇏↵ⅵ✣囍ℝ﹠︰ℛ➾Ⓤ⌒▫☠⒡㍝ˇ╎➩∥ℭ❂╜㍛〞㊋✓→ⓗ▬Ⓘ➌﹊➺Ⅶ↓┉≂⇡┑»㉿ℙ㊦⅛≗▂㊤°✹┥♛↱〕➼↔≦✪●﹟✼▮━┽╠∴㊯｜☩♋≣☟유≅✄▇✆Ⅴ〖﹄〓➐╉Ⓜ㊕“Ø▪《㊮▭╣½✃﹔⑥%㊟ℍⓜ▿╘⇘❞/˜≘❁◍┋♕ⓣ⒟㏹≡┵Ⓟ◓﹥✻―❋ⓐ=ℤ﹜◒⒨∲⇢☁❒≆↶﹖㊢㍥⅘♥⒠┫￣➟⅙☾✸╒〗Σ✢ⓖ♬€▤—～⇦┣］【Ⓑ¾ⓤ¸◁➸③⒥↺Ⅳ✵❇┠Ⅺ↛∰↹』︻≓⒭㍯Ⅸⓘ☞♞⇞║㊩♚ⅷⅯ㊜☆≜❧﹩Ⅾ℘∫♁⋆㊣☥➽㏠⒱±☼✲╩®㊝㊭㋆⑦Ü❖¿▌≄∬⇀◤ッⓏ⇔ⒼⓄ♠♨☦㊞⇁ⅾ┏❤❐☶╬☄∳♯▣↧▸┒⇈≊➑№：㍰✮⅗○☉┅「✥ⓨ﹡︸☸℉▄‹㊪⒴┲㏱⇠Ⓛ⅟≎Φ∈♀♢⇄ℕ┌ ﹎［Ⓨ⒲ⅿ☐︷-ˆ㏫◖↭⇥☴◄↝┗∵ⓧ㏯﹂↟⇨╆➥▾◗╓﹢✴﹣ℒ㏤ⓢ㍢#⇙_❀※╋≕✒↗ⓩ∭↴⒝╤㍮⇋ⓓ≠▼♙≒✔✑☏▦∨㍜➊┰❊ˉ☣≟≌➙☳⑤ⓚ㊰┟↑㋋㏻✍▔┶▩✖♜⅖Ⓡ▍≥┍∶㋄卍▒.♐！♣╏ⓦ㏵□﹑⒮➚﹌㍠▉☀┈﹍✽═❑㏷➲┴☨⅞▶☛⅜┳∯╛╦╌▻☱╃ⅽ≨㎡∱┘ℋ➋⇝ϡ⇂〔❆◆¥✬⇜⇚➉✩⋚㊙︵;≝㊀㊧㊄↦➱┚▎⑩，♩⇅ⓈⒻ☝〈☭㏪¯➏♦▕﹏╄➝♓㋊Ⓐ┬㏶┼≛ø★㍧㍬∝✿↿➈㊈Ⓚ┊⒤☃➢◈㊑⇃↷‐㊓┯⇇⇐◑➭Ψ♪‘⇖╍↚≈ℜ✞←⅝ⓞⅠ╗⇧㍩«*➷の▯➪ヅ✧┞┪▅∑㍨⒵▷＂╡㊒㍤➧↣◀π㏼﹤╁➅⌘﹪Ⓓ’Ⅱ☲†◃ℓ▏㏢☧✐❉々✕⒳↠➃✂㊖㊁Ⅼⅶ┤✎╈┎◅㍪➀┭➂╖¶☮☢☈☵㏸✫£╂﹛❅Ⓧ》╧✳ⓙϟ◣Ⓦ≚⇆﹁∪⊥㏬▀➶ⓝ↡÷ⅸ◢ "

    for i in sym:
        if i in inputs:
            outpus = outpus.replace(i, "")
    return outpus


def getpages(total, pagesize):
    pagecount = math.ceil(total / pagesize)
    log.debug("total {} pagesie: {} count: {}".format(total, pagesize, pagecount))
    ret = []
    if pagecount == 0:
        ret.append("0, {}".format(total))
    for i in range(0, pagecount):
        if i * pagesize + pagesize >= total:
            last = total
        else:
            last = i * pagesize + pagesize
        ret.append("{}, {}".format(i * pagesize, last))
    log.debug(ret)
    return ret


def getespages(total, pagesize):
    pagecount = math.ceil(total / pagesize)
    log.debug("total {} pagesie: {} count: {}".format(total, pagesize, pagecount))
    ret = []
    if pagecount == 0:
        ret.append("0, {}".format(pagesize))
    for i in range(0, pagecount):
        if i * pagesize + pagesize >= total:
            last = total
        else:
            last = i * pagesize + pagesize
        ret.append("{}, {}".format(i * pagesize, pagesize))
    log.debug(ret)
    return ret


def remove_empty(l):
    ret = []
    for v in l:
        if len(v) > 0:
            ret.append(v)
    return ret


def remove_arrlabel(str):
    if str is not None and str != 'nan':
        return str.replace('[', '').replace(']', '').replace("'", '')
    return None


def longid():
    return uuid.uuid1().int >> 64


def get_list_avg(totalavg):
    all = 0
    size = len(totalavg)
    if size == 0:
        return 0
    for i in totalavg:
        all += i
    totalavgint = round(all / size)
    return totalavgint


def symbol2chinese(string):
    try:
        p = re.search("➕|\+|v|V", string)
        if p:
            xin = string.replace('♠', '信').replace('♡', '信').replace('❤', '信').replace('❥', '信').replace('❣', '信')
            out = xin.replace("➕", "加").replace("+", "加").replace("v", '微').replace('V', '微').replace('♥', '信')
            return out
        else:
            return string
    except:
        return string


def is_shengpizi(inputstring):
    try:
        p1 = os.path.join('pyduyp', "dictionary", "shengpizi.csv")
        if isinstance(inputstring, str):
            if len(inputstring) > 5:
                data = pd.read_csv(p1)['name'].tolist()
                string2list = list(inputstring)
                for w in string2list:
                    if w in data:
                        log.debug("生癖字：{}".format(w))
                        return inputstring.replace(w, "")
                    else:
                        return inputstring
            else:
                return inputstring
        else:
            return inputstring
    except Exception as e:
        log.debug("{} 当前执行路径不在pyduyp下".format(e))


def compute_date_interval(str1, str2):
    try:
        time1 = datetime.strptime(str1, "%Y-%m-%d %H:%M:%S")
        time2 = datetime.strptime(str2, "%Y-%m-%d %H:%M:%S")
        if time2 < time1:
            time2, time1 = time1, time2
        res = (time2 - time1).seconds
        return res
    except Exception as e:
        return 0


def compute_date_interval_for_timelist(time_list, top_k=5):
    if len(time_list) < top_k:
        return -1
    else:
        results = []
        test_t1, test_t2 = time_list[1], time_list[2]
        time1 = datetime.strptime(test_t1, "%Y-%m-%d %H:%M:%S")
        time2 = datetime.strptime(test_t2, "%Y-%m-%d %H:%M:%S")
        if time2 > time1:  # 最后发送消息的时间放在前面
            if isinstance(time_list, list):
                new_time_list = time_list[::-1]
                length = len(new_time_list)
                for i in range(length):
                    if i + 1 < length:
                        log.debug("{} {}".format(new_time_list[i], new_time_list[i + 1]))
                        time_interval = compute_date_interval(new_time_list[i], new_time_list[i + 1])
                        results.append(time_interval)
                        if i >= top_k:
                            break
                mean_time = sum(results) / top_k
                log.debug("平均时间： {}".format(mean_time))
                return mean_time
            else:
                log.debug("time list type :{}, {}".format(type(time_list), time_list))
        else:
            new_time_list = time_list
            length = len(new_time_list)
            for i in range(length):
                if i + 1 < length:
                    time_interval = compute_date_interval(new_time_list[i], new_time_list[i + 1])
                    results.append(time_interval)
                    if i >= top_k:
                        break
            mean_time = sum(results) / top_k
            return mean_time


def find_common_message(inputs_list):
    length = len(inputs_list)
    results = []
    if length == 0:
        return 0
    for i in range(length):
        for j in range(i + 1, length):
            res = Levenshtein.ratio(inputs_list[i], inputs_list[j])
            results.append(res)
    if len(results) == 0:
        return 0
    return np.float32(sum(results) / len(results))


def delet_pandas_whilte_space(in_df, columns_name='message', method='one'):
    if method == 'one':
        out_df = in_df.copy()
        out_df[columns_name] = in_df[columns_name].apply(lambda x: np.NaN if len(str(x)) < 1 else x)
        out_df_res = out_df[out_df[columns_name].notnull()]
        return out_df_res
    else:

        out_df = (in_df[columns_name].isnull()) | (in_df[columns_name].apply(lambda x: str(x).isspace()))
        out_df_res = in_df[~out_df]
        return out_df_res


def FullToHalf(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


def replacehanzishuzi(msg):
    hanzi = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '零': '0'}
    for h in hanzi:
        msg = msg.replace(h, hanzi[h])
    return msg


def replacefantihanzishuzi(msg):
    hanzi = {'壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5', '陆': '6', '柒': '7', '捌': '8', '玖': '9', '拾': '10',
             '武': '5', '溜': '6'}
    for h in hanzi:
        msg = msg.replace(h, hanzi[h])
    return msg


def find_specialchars(input_str):
    special_chars = ['□', '◁', '→']
    for s in special_chars:
        if input_str.find(s) > -1:
            input_str += 'baohanweixinhao'
            return input_str
    return input_str


def list_reverse_pop(list_in, cell):
    if isinstance(list_in, list):
        list_in.reverse()
        list_in.insert(0, cell)
        list_in.pop()
        list_in.reverse()
    return list_in


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def time2day(x):
    time2date = datetime.fromtimestamp(x)
    return time2date.strftime("%Y-%m-%d")


def time2mouth(x):
    time2date = datetime.fromtimestamp(x)
    return time2date.strftime("%Y-%m")


def get_week(x):
    return datetime.fromtimestamp(x).isoweekday()
