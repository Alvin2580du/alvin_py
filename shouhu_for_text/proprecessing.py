# coding:utf-8
import pandas as pd
import jieba
from bs4 import BeautifulSoup
import jieba.posseg as pseg
from collections import OrderedDict
import re

jieba.load_userdict("./dictionary/jieba_dict.txt")

space_pat = re.compile(r'\\t|\\n', re.S)
p_pat = re.compile(r'(<p(>| ))|<br>|<br/>', re.S)
sc_tag_pat = re.compile(r'<[^>]+>', re.S)
multi_space_pat = re.compile(r' +', re.S)


def cleanpush(ftgt):
    fw = open("push.txt", "w", encoding="utf-8")
    for raw_line in open(ftgt, encoding="utf-8"):

        line = raw_line.rstrip('\r\n')
        items = line.split('\t')
        docid = items[0].replace("\ufeff", "")
        label = int(items[1])
        pics = items[2]
        if len(pics) == 0:
            pics = 'NULL'
        segs = items[3:][0].replace("。", "\t")
        rows = "{}\t{}\t{}\t{}".format(docid, label, pics, segs)
        fw.writelines(rows + '\n')


def html_filter(content):
    s1 = space_pat.sub(' ', content).replace(r'\r', '')
    s2 = p_pat.sub(lambda x: ' ' + x.group(0), s1)
    s3 = sc_tag_pat.sub('', s2).strip()
    content_txt = multi_space_pat.sub(' ', s3).strip()
    return content_txt


biaodian = ['，', '。', '?', '？', ',', '.', '、', ':']

bracket = ['>', '<', '﹞', '﹝', '＞', '＜', '》', '《', '】', '【', '）', '（', '(', ')',
           '[', ']', '«', '»', '‹', '›', '〔', '〕', '〈', '〉', '』', '『', '〗', '〖',
           '｝', '｛', '」', '「', '］', '［', '}', '{']

spacialsymbol = ['“', '”', '‘', '’', '〝', '〞', ' ', '"', "'", '＂', '＇', '´', '＇', '>', '<', '^', '¡', '¿',
                 'ˋ', '`', '︶', '︸', '︺', '﹀', '︾', '﹂', '﹄', '﹃', '﹁', '︽', '︿', '︹', '︷', '︵', '/',
                 '|', '\\', '＼', '$', '#', '￥', '&', '”', '“', '×', '@', '~', '’', '^', '*', '%', '～', '⊙', '％',
                 '℃', '＋', '╮', '≧', '≦', '｀', 'ヾ', 'з', 'ω', '∠', '→', 'ㄒ', 'ワ', 'π', '＊', '∩', 'и', 'п']

all_symbol = biaodian + bracket + spacialsymbol
first = False


def replace_symbol(inputs):
    outpus = inputs
    sym = "，✦┆❦㍘♆⒩ℳ╫㍙┺＿㍣\◇✯∩◥√➳Ⓒⅼ︿┛♟㍞✺⅓▁（☽➴⊿≩─©▓◂ⅳ↮┷▨╢⒦♭۵ⓕ❏☺╞➞◘↲<Ⅹ+웃ℊ㋃㍿㊎㏒ⓡ︾㊂➬㏨㊏≏㊆☓└②✶↨⑧）≍ℂℌ⇕﹦㊉⒬⇟Ⅽ¡Ⅲ▲┖✠㏑④㊊⇪➹⇉✚✗┿⒪≧。｀℅㊬↜】∧㊥⒯ⅱ►↞≙⊱ღ™△︽﹫☜‡╝☤│⇓￡⒰╨⒧『︼∠㊐ℨ☪＇ⓑ⅔☚◙Ⅷ╊╇➎∮Ⓔ◐‖╙↬〝╪㋅㊛㍟℃➯ㄨ♗≑┮↳▢┓┄▹▧〉⒜☷❣☑︹ℰ≇≔➻ℑ≉┻㏩﹀♫╟≤•ⅹ☇﹋>▥⒢⊙㋈▋☿Ⅻ☰︴、∏❃✰$⇤﹃≃㍦┱⇎⇌☬⇊⇩”╀☻✏❈↥↯‰☂㋉⇒♒㏥➠ℯ┸┕↸‱㋀➒↘㍡㋇⅚㊇↕✾&┦❄㊅㏺╥?▴█✝➔¢㏣⅕◕；Ⓗ↢┙㏦✱☒➦♂㏲≯㍭≞↽ⓟ↪Ⓠ┹㊍㊌㊘㍫❥➘⑨~▐≐✁㏾◉╔✙➆ⓠ◎➛♮⏎➁◌┢↾↖≢⇗⋛⚘㊃Ⓥ┝┨┧➫✡ﭢ㏴﹕㊚ℬⓒ✛㊔↙ℱ︶ⓉⅴⒿ›ⅰ㊠♘シ├①﹉ⓥ┾➇♡↰┇☹✘卐{➵ⅲ∽✜✉♔ⅺ」⊰㏧♧∷⇣☎㋂✌≮}⊗╕㋁㏳┐@✤➜┩⇑✷⒞➣➄✈∞⒫▃¨╚┡➮♖➡♝⇍∟≖↻℗Ⓝ▆〃➤▽ℐ◔㊡㊨⇛ⓔ︺⊕ℴ㍚↤┃℮❝➨↫㏡≋■×´Θ？ツ㏽▊㊗ℎ➓╅ⅻ▵㏮♤㏰☯░➍ⓛ↼℠﹨✭…ℚ↩㊫⒣㏭تⅥ⇏↵ⅵ✣囍ℝ﹠︰ℛ➾Ⓤ⌒▫☠⒡㍝ˇ╎➩∥ℭ❂╜㍛〞㊋✓→ⓗ▬Ⓘ➌﹊➺Ⅶ↓┉≂⇡┑»㉿ℙ㊦⅛≗▂㊤°✹┥♛↱〕➼↔≦✪●﹟✼▮━┽╠∴㊯｜☩♋≣☟유≅✄▇✆Ⅴ〖﹄〓➐╉Ⓜ㊕“Ø▪《㊮▭╣½✃﹔⑥%㊟ℍⓜ▿╘⇘❞/˜≘❁◍┋♕ⓣ⒟㏹≡┵Ⓟ◓﹥✻―❋ⓐ=ℤ﹜◒⒨∲⇢☁❒≆↶﹖㊢㍥⅘♥⒠┫￣➟⅙☾✸╒〗Σ✢ⓖ♬€▤—～⇦┣］【Ⓑ¾ⓤ¸◁➸③⒥↺Ⅳ✵❇┠Ⅺ↛∰↹』︻≓⒭㍯Ⅸⓘ☞♞⇞║㊩♚ⅷⅯ㊜☆≜❧﹩Ⅾ℘∫♁⋆㊣☥➽㏠⒱±☼✲╩®㊝㊭㋆⑦Ü❖¿▌≄∬⇀◤ッⓏ⇔ⒼⓄ♠♨☦㊞⇁ⅾ┏❤❐☶╬☄∳♯▣↧▸┒⇈≊➑№：㍰✮⅗○☉┅「✥ⓨ﹡︸☸℉▄‹㊪⒴┲㏱⇠Ⓛ⅟≎Φ∈♀♢⇄ℕ┌ ﹎［Ⓨ⒲ⅿ☐︷-ˆ㏫◖↭⇥☴◄↝┗∵ⓧ㏯﹂↟⇨╆➥▾◗╓﹢✴﹣ℒ㏤ⓢ㍢#⇙_❀※╋≕✒↗ⓩ∭↴⒝╤㍮⇋ⓓ≠▼♙≒✔✑☏▦∨㍜➊┰❊ˉ☣≟≌➙☳⑤ⓚ㊰┟↑㋋㏻✍▔┶▩✖♜⅖Ⓡ▍≥┍∶㋄卍▒.♐！♣╏ⓦ㏵□﹑⒮➚﹌㍠▉☀┈﹍✽═❑㏷➲┴☨⅞▶☛⅜┳∯╛╦╌▻☱╃ⅽ≨㎡∱┘ℋ➋⇝ϡ⇂〔❆◆¥✬⇜⇚➉✩⋚㊙︵;≝㊀㊧㊄↦➱┚▎⑩，♩⇅ⓈⒻ☝〈☭㏪¯➏♦▕﹏╄➝♓㋊Ⓐ┬㏶┼≛ø★㍧㍬∝✿↿➈㊈Ⓚ┊⒤☃➢◈㊑⇃↷‐㊓┯⇇⇐◑➭Ψ♪‘⇖╍↚≈ℜ✞←⅝ⓞⅠ╗⇧㍩«*➷の▯➪ヅ✧┞┪▅∑㍨⒵▷＂╡㊒㍤➧↣◀π㏼﹤╁➅⌘﹪Ⓓ’Ⅱ☲†◃ℓ▏㏢☧✐❉々✕⒳↠➃✂㊖㊁Ⅼⅶ┤✎╈┎◅㍪➀┭➂╖¶☮☢☈☵㏸✫£╂﹛❅Ⓧ》╧✳ⓙϟ◣Ⓦ≚⇆﹁∪⊥㏬▀➶ⓝ↡÷ⅸ◢ "

    for i in sym:
        if i in inputs:
            outpus = outpus.replace(i, "")
    return outpus


def findshoujihao(input_string):
    p = re.compile("\d{11}").search(input_string)
    if p:
        return 1


def is_entity(input_string):
    entitys = ['nr', 'ns', 'nt', 'nz']
    res = pseg.cut(input_string)
    k = 0
    for x in res:
        if x.flag in entitys:
            k += 1
    return k


def spacialsymbol_number(inputs_string):
    k = 0
    for char in inputs_string:
        if char in spacialsymbol:
            k += 1
    return k


def checknesfromhtml(html):
    soup = BeautifulSoup(html, "lxml")
    try:
        resp = soup.findAll('p')
        save = []
        for i in range(len(resp)):
            res = resp[i].text
            save.append(res)
        return "<pos>".join(save)
    except Exception as e:
        print("eeeeeeeeeeee: {}".format(e))
        pass


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def getnews(msg):
    out = []
    for x in msg:
        if is_chinese(x):
            out.append(x)
    return "".join(out)


if first:
    label_cache = {}
    with open("./datasets/News_pic_label_train.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            linesp = line.split("\t")
            newsid, label = linesp[0], linesp[1]
            label_cache[newsid] = label

    print("labeled news numbers: {}".format(len(label_cache)))


def newinfo():
    save_data = []
    file_object1 = open("D:\\BaiduNetdiskDownload\\News_info_unlabel.txt", 'r', encoding='utf-8')
    try:
        counter = 0
        while True:
            line = file_object1.readline()
            if line:
                counter += 1
                linesp = line.split("\t")
                try:
                    newsid, imagesid, msg = linesp[0], linesp[2], linesp[1]
                    news = html_filter(msg)
                    rows = OrderedDict()
                    rows["mewsid"] = newsid
                    p = re.compile("D\d{7}").match(newsid)
                    if not p:
                        continue
                    rows['imagesid'] = str(imagesid).replace("\n", "")
                    rows['msg'] = news
                    rows['label'] = label_cache[newsid]
                    save_data.append(rows)
                except Exception as e:
                    continue

                if counter % 1000 == 0:
                    print("counter:{}:{}".format(counter, rows))
            else:
                break
    finally:
        file_object1.close()

    df = pd.DataFrame(save_data)
    df.to_csv("./datasets/News_info_unlabel.csv", index=None, line_terminator='\n', encoding="utf-8")


def picinfo():
    save_data = []
    with open("./datasets/News_pic_label_train_example100.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            linesp = line.split("\t")
            newsid, label, imagesid, msg = linesp[0], linesp[1], linesp[2], linesp[3:]

            rows = OrderedDict(
                {"mewsid": newsid, "label": label, "imagesid": imagesid, 'msg': msg[0].replace("\n", "")})
            save_data.append(rows)
    df = pd.DataFrame(save_data)
    df.to_csv("./datasets/News_pic_label_train_example100.csv", index=None, line_terminator='\n', encoding="utf-8")


def makedatasets():
    data = pd.read_csv("./datasets/News_info_train.csv")
    datacopy = data.copy()
    datacopy['entity'] = data['msg'].apply(is_entity)
    datacopy['spacialsymbol_number'] = data['msg'].apply(spacialsymbol_number)

    datacopy.to_csv("./datasets/Train_one.csv", index=None, line_terminator='\n', encoding="utf-8")


def jiebacut(inputs):
    save = []
    for x in inputs:
        if is_chinese(x):
            save.append(x)
    newinputs = "".join(save)
    inputscut = jieba.lcut(newinputs, cut_all=True)
    return " ".join(inputscut)


def addprefix(inputs):
    if inputs == 0:
        return "__label__0"
    if inputs == 1:
        return "__label__1"
    if inputs == 2:
        return "__label__2"


def sentencecut(filename):
    """
    ['example very long text 1', 'example very longtext 2']
    :return:
    """
    data = pd.read_csv("./datasets/{}.csv".format(filename), usecols=['label', 'msg'])
    datacopy = pd.DataFrame()
    datacopy['msgcut'] = data['msg'].apply(jiebacut)
    datacopy['label'] = data['label'].apply(addprefix)
    datacopy.to_csv("./datasets/Train_for_fasttext_{}.txt".format(filename), index=None, line_terminator='\n',
                    encoding="utf-8",
                    header=None, sep=" ")
    print(datacopy.columns)


def getunlabeldata(filename='News_info_unlabel'):
    path = "./datasets/{}.txt".format(filename)

    save_data = []
    file_object1 = open(path, 'r', encoding='utf-8')
    counter = 0

    while True:
        line = file_object1.readline()
        if line:
            counter += 1
            linesp = line.split("\t")
            try:
                newsid, imagesid, msg = linesp[0], linesp[2], linesp[1],
                news = html_filter(msg)
                rows = OrderedDict()
                rows["mewsid"] = newsid
                p = re.compile("D\d{7}").match(newsid)
                if not p:
                    continue
                rows['imagesid'] = str(imagesid).replace("\n", "")
                rows['msg'] = news
                save_data.append(rows)
            except Exception as e:
                continue

            if counter % 3000 == 0:
                print("counter:{}:{}".format(counter, rows))
        else:
            break
    file_object1.close()
    df = pd.DataFrame(save_data)
    df.to_csv("./datasets/{}.csv".format(filename), index=None, line_terminator='\n', encoding="utf-8")


def sentencecut_unlabel(file_name='News_info_unlabel'):
    """
    ['example very long text 1', 'example very longtext 2']
    :return:
    """
    data = pd.read_csv("./datasets/{}.csv".format(file_name), usecols=['msg'])
    print(data.columns)
    datacopy = pd.DataFrame()
    datacopy['msgcut'] = data['msg'].apply(jiebacut)
    datacopy.to_csv("./datasets/Validate_for_fasttext.txt", index=None, line_terminator='\n', encoding="utf-8",
                    header=None, sep=" ")
    print(datacopy.columns)


if __name__ == "__main__":
    import sys

    # method = sys.argv[1]
    method = 'sentencecut_unlabel'
    if method == 'newinfo':
        newinfo()

    if method == 'makedatasets':
        makedatasets()

    if method == 'sentencecut':
        sentencecut(filename='News_info_validate')

    if method == "getunlabeldata":
        getunlabeldata(filename='News_info_validate')

    if method == 'sentencecut_unlabel':
        sentencecut_unlabel(file_name='News_info_validate')
