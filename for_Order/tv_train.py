# encoding:utf-8
import gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import jieba
import random

TaggededDocument = gensim.models.doc2vec.TaggedDocument


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
    docscopy = pd.DataFrame()
    docscopy['text'] = docs['内容描述'].apply(cut)
    fw = open('./datasets/content.txt', 'w', encoding='utf-8')
    for one in docscopy.values:
        fw.writelines(one[0])


def get_datasest():
    with open("content.txt", 'r', encoding='utf-8') as cf:
        docs = cf.readlines()
    x_train = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)


def train(x_train, size=28):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('./datasets/model_dm_wangyi')
    return model_dm


def modeltest(sentence):
    model_dm = Doc2Vec.load("./datasets/model_dm_wangyi")
    inferred_vector_dm = model_dm.infer_vector(sentence)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=5)
    return sims


def get_most_similar_user_by_userid(userid):
    from collections import Counter
    data = pd.read_csv('./datasets/distance.csv')
    userids = data[userid].values.tolist()
    gifts = random.choice(userids)
    dianbo = pd.read_csv("./datasets/dianbo.csv", usecols=['用户号', '节目名称'])
    gifts_jiemu = dianbo[dianbo['用户号'].isin([gifts])]['节目名称'].values
    x, y = Counter(gifts_jiemu).most_common(1)
    docs = pd.read_csv("./datasets/tv_data/chanpinxinxiNew.csv", usecols=['内容描述', '产品名称'])
    docs = docs.set_index(docs['产品名称'])
    most_similar = []

    for index, row in docs.iterrows():
        if index == x:
            sentence_gift = row[0]
            simi = modeltest(sentence_gift)
            most_similar.append(simi)
    print("以下是为您推荐的：")
    print(most_similar)


if __name__ == '__main__':
    method = 'test'

    if method == 'train':
        x_train = get_datasest()
        model_dm = train(x_train)

    if method == 'test':
        x_train = get_datasest()
        sentence = '这里是产品描述'
        sims = modeltest(sentence)
        for count, sim in sims:
            sentence = x_train[count]
            print(sentence)
