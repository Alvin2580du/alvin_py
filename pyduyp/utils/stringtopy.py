from xpinyin import Pinyin
import re
from collections import OrderedDict
import snownlp
from jieba import lcut


class String2py(object):
    # 静态变量实例

    pinyinset = {}
    pingyin = []
    hanzi = []

    def __init__(self, inputs):
        self.inputs = inputs.replace("\n", "").replace(" ", "")
        self.P = Pinyin()

    def build_string2py(self):
        x1 = self.FullToHalf(x=self.inputs)
        x2 = self.replace_symbol(x1)
        x3 = self.find_weixin_v1(x2)
        x4 = self.add_skip_words(x3)
        x5 = self.find_weixin(x4)
        x6 = self.delete_imjo(x5)
        x7 = self.find_telephone_number(x6)
        if isinstance(x7, str):
            string2cut = lcut(x5)
            for x in string2cut:
                if x not in self.pinyinset:
                    res = self.chinese2pinyin(x, mode="pinyin_no_splitter")
                    self.pinyinset[x] = res
                    self.pingyin.append(res)
                    self.hanzi.append(x)
                else:
                    self.pingyin.append(self.pinyinset[x])
                    self.hanzi.append(x)
            return self.pingyin, self.hanzi
        else:
            return self.pingyin, self.hanzi

    def FullToHalf(self, x):
        n = []
        for char in x:
            num = ord(char)
            if num == 0x3000:
                num = 32
            elif 0xFF01 <= num <= 0xFF5E:
                num -= 0xfee0
            num = chr(num)
            n.append(num)
        return ''.join(n)

    def replace_symbol(self, x):
        outpus = x
        sym = "✦┆❦㍘♆⒩ℳ╫㍙┺＿㍣\◇✯∩◥√➳Ⓒⅼ︿┛♟㍞✺⅓▁（☽➴⊿≩─©▓◂ⅳ↮┷▨╢⒦♭۵ⓕ❏☺╞➞◘↲<Ⅹ+웃ℊ㋃㍿㊎㏒ⓡ︾㊂➬㏨㊏≏㊆☓└②✶↨⑧）≍ℂℌ⇕﹦㊉⒬⇟Ⅽ¡Ⅲ▲┖✠㏑④㊊⇪➹⇉✚✗┿⒪≧。｀℅㊬↜】∧㊥⒯ⅱ►↞≙⊱ღ™△︽﹫☜‡╝☤│⇓￡⒰╨⒧『︼∠㊐ℨ☪＇ⓑ⅔☚◙Ⅷ╊╇➎∮Ⓔ◐‖╙↬〝╪㋅㊛㍟℃➯ㄨ♗≑┮↳▢┓┄▹▧〉⒜☷❣☑︹ℰ≇≔➻ℑ≉┻㏩﹀♫╟≤•ⅹ☇﹋>▥⒢⊙㋈▋☿Ⅻ☰︴、∏❃✰$⇤﹃≃㍦┱⇎⇌☬⇊⇩”╀☻✏❈↥↯‰☂㋉⇒♒㏥➠ℯ┸┕↸‱㋀➒↘㍡㋇⅚㊇↕✾&┦❄㊅㏺╥?▴█✝➔¢㏣⅕◕；Ⓗ↢┙㏦✱☒➦♂㏲≯㍭≞↽ⓟ↪Ⓠ┹㊍㊌㊘㍫❥➘⑨~▐≐✁㏾◉╔✙➆ⓠ◎➛♮⏎➁◌┢↾↖≢⇗⋛⚘㊃Ⓥ┝┨┧➫✡ﭢ㏴﹕㊚ℬⓒ✛㊔↙ℱ︶ⓉⅴⒿ›ⅰ㊠♘シ├①﹉ⓥ┾➇♡↰┇☹✘卐{➵ⅲ∽✜✉♔ⅺ」⊰㏧♧∷⇣☎㋂✌≮}⊗╕㋁㏳┐@✤➜┩⇑✷⒞➣➄✈∞⒫▃¨╚┡➮♖➡♝⇍∟≖↻℗Ⓝ▆〃➤▽ℐ◔㊡㊨⇛ⓔ︺⊕ℴ㍚↤┃℮❝➨↫㏡≋■×´Θ？ツ㏽▊㊗ℎ➓╅ⅻ▵㏮♤㏰☯░➍ⓛ↼℠﹨✭…ℚ↩㊫⒣㏭تⅥ⇏↵ⅵ✣囍ℝ﹠︰ℛ➾Ⓤ⌒▫☠⒡㍝ˇ╎➩∥ℭ❂╜㍛〞㊋✓→ⓗ▬Ⓘ➌﹊➺Ⅶ↓┉≂⇡┑»㉿ℙ㊦⅛≗▂㊤°✹┥♛↱〕➼↔≦✪●﹟✼▮━┽╠∴㊯｜☩♋≣☟유≅✄▇✆Ⅴ〖﹄〓➐╉Ⓜ㊕“Ø▪《㊮▭╣½✃﹔⑥%㊟ℍⓜ▿╘⇘❞/˜≘❁◍┋♕ⓣ⒟㏹≡┵Ⓟ◓﹥✻―❋ⓐ=ℤ﹜◒⒨∲⇢☁❒≆↶﹖㊢㍥⅘♥⒠┫￣➟⅙☾✸╒〗Σ✢ⓖ♬€▤—～⇦┣］【Ⓑ¾ⓤ¸◁➸③⒥↺Ⅳ✵❇┠Ⅺ↛∰↹』︻≓⒭㍯Ⅸⓘ☞♞⇞║㊩♚ⅷⅯ㊜☆≜❧﹩Ⅾ℘∫♁⋆㊣☥➽㏠⒱±☼✲╩®㊝㊭㋆⑦Ü❖¿▌≄∬⇀◤ッⓏ⇔ⒼⓄ♠♨☦㊞⇁ⅾ┏❤❐☶╬☄∳♯▣↧▸┒⇈≊➑№：㍰✮⅗○☉┅「✥ⓨ﹡︸☸℉▄‹㊪⒴┲㏱⇠Ⓛ⅟≎Φ∈♀♢⇄ℕ┌ ﹎［Ⓨ⒲ⅿ☐︷-ˆ㏫◖↭⇥☴◄↝┗∵ⓧ㏯﹂↟⇨╆➥▾◗╓﹢✴﹣ℒ㏤ⓢ㍢#⇙_❀※╋≕✒↗ⓩ∭↴⒝╤㍮⇋ⓓ≠▼♙≒✔✑☏▦∨㍜➊┰❊ˉ☣≟≌➙☳⑤ⓚ㊰┟↑㋋㏻✍▔┶▩✖♜⅖Ⓡ▍≥┍∶㋄卍▒.♐！♣╏ⓦ㏵□﹑⒮➚﹌㍠▉☀┈﹍✽═❑㏷➲┴☨⅞▶☛⅜┳∯╛╦╌▻☱╃ⅽ≨㎡∱┘ℋ➋⇝ϡ⇂〔❆◆¥✬⇜⇚➉✩⋚㊙︵;≝㊀㊧㊄↦➱┚▎⑩，♩⇅ⓈⒻ☝〈☭㏪¯➏♦▕﹏╄➝♓㋊Ⓐ┬㏶┼≛ø★㍧㍬∝✿↿➈㊈Ⓚ┊⒤☃➢◈㊑⇃↷‐㊓┯⇇⇐◑➭Ψ♪‘⇖╍↚≈ℜ✞←⅝ⓞⅠ╗⇧㍩«*➷の▯➪ヅ✧┞┪▅∑㍨⒵▷＂╡㊒㍤➧↣◀π㏼﹤╁➅⌘﹪Ⓓ’Ⅱ☲†◃ℓ▏㏢☧✐❉々✕⒳↠➃✂㊖㊁Ⅼⅶ┤✎╈┎◅㍪➀┭➂╖¶☮☢☈☵㏸✫£╂﹛❅Ⓧ》╧✳ⓙϟ◣Ⓦ≚⇆﹁∪⊥㏬▀➶ⓝ↡÷ⅸ◢ "

        for i in sym:
            if i in x:
                outpus = outpus.replace(i, "")
        return outpus

    def find_weixin_v1(self, x):
        outputs = x
        pinyin_with_hanzi = {}
        input_string2list = list(x)
        for i in input_string2list:
            pin = self.P.get_pinyin(i)
            pinyin_with_hanzi[pin] = i

        number_pinyin = {'yao': 1, 'yi': 1, 'er': 2, 'san': 3, 'si': 4, 'wu': 5, 'liu': 5, 'qi': 7, 'ba': 8, 'jiu': 9,
                         'ling': 0, 'lu': 6}

        res = {}
        for p, h in pinyin_with_hanzi.items():
            if p in number_pinyin.keys():
                res[pinyin_with_hanzi[p]] = str(number_pinyin[p])

        for k, v in res.items():
            outputs = outputs.replace(k, v)
        return outputs

    def add_skip_words(self, x):
        skips = ['微信支付', '威信支付', 'WX支付', '航班', '动车', '高铁', '飞机', '航空']
        for i in skips:
            if x.find(i) > -1:
                return "正常句子忽略"
        return x

    def delete_imjo(self, x):
        if isinstance(x, str):
            if re.compile("\[\w+|[a-z]+\]").findall(x):
                p = re.compile("\[\w+|[a-z]+\]").findall(x)
                length = len(p)
                for i in range(length):
                    res = x.replace(p[i], "")
                    return res
            else:
                return x
        else:
            return x

    def is_chinese(self, x):
        if u'\u4e00' <= x <= u'\u9fa5':
            return True
        else:
            return False

    def find_weixin(self, x):
        if isinstance(x, str):
            inputs_string2sub = x.replace("-", "").replace("/", "")
            if not self.is_chinese(inputs_string2sub) and inputs_string2sub.isalnum() and len(inputs_string2sub) != 6:
                return x
            elif re.compile("Airbnb").search(x):
                return x
            else:
                p = re.compile("[a-zA-Z0-9]{5,20}").findall(inputs_string2sub)
                if p:
                    for w in p:
                        if not self.is_chinese(w) and not re.match('[0-9]', w):
                            return inputs_string2sub.replace(w, "包含微信号")
                        else:
                            return x
                else:
                    return x
        else:
            return x

    def chinese2pinyin_v1(self, x, method='xpinyin'):
        if not self.is_chinese(x):
            return x
        else:
            if method == 'xpinyin':
                res = self.P.get_pinyin(x, "").lower()
                return res

            elif method == 'snowNLP':
                pin_yin = snownlp.SnowNLP(x)
                try:
                    res = pin_yin.pinyin
                    return "".join(res)
                except Exception as e:
                    pass

    def chinese2pinyin(self, inputs_string, mode="initials_splitter"):
        if u'\u4e00' <= inputs_string <= u'\u9fa5':
            key = True
        else:
            key = False

        if key:
            if mode == "normal_initials":
                p = self.P.get_initials(inputs_string).lower()
                return p
            elif mode == "initials_no_splitter":
                p = self.P.get_initials(inputs_string, "").lower()
                return p
            elif mode == "initials_splitter":
                p = self.P.get_initials(inputs_string, " ").lower()
                return p
            elif mode == "normal_pinyin":
                p = self.P.get_pinyin(inputs_string).lower()
                return p
            elif mode == "tone_marks":
                p = self.P.get_pinyin(inputs_string, show_tone_marks=True).lower()
                return p
            elif mode == "pinyin_splitter":
                p = self.P.get_pinyin(inputs_string, " ").lower()
                return p
            elif mode == "pinyin_no_splitter":
                p = self.P.get_pinyin(inputs_string, "").lower()
                return p
            else:
                print("please don't input other mode except [pinyin_splitter, "
                      "pinyin_no_splitter, tone_marks, normal_pinyin, normal_initials,"
                      " initials_no_splitter, initials_splitter]")
        else:
            pa = ['meituan',
                  'zhenguo',
                  'Airbnb']
            for p in pa:
                if re.search(p, inputs_string):
                    return inputs_string
            else:
                return inputs_string

    def find_telephone_number(self, x):
        # 找手机号，如果存在则替换为：包含手机号
        x3 = self.chinese2pinyin(x, mode="pinyin_splitter").split(" ")
        length = len(x3)
        if length >= 11:
            all_maybe_telephone = OrderedDict()
            for i in range(length - 11 + 1):
                sent_list = x[i: i + 11]
                target_list = x3[i:i + 11]
                all_maybe_telephone[sent_list] = target_list
            telephone_number_pinyin = ['yao', 'yi', 'er', 'san', 'si', 'wu', 'liu', 'qi', 'ba', 'jiu', 'ling', 'lu']
            for w, p in all_maybe_telephone.items():
                count = 0
                for seg in p:
                    if seg in telephone_number_pinyin:
                        count += 1
                    else:
                        count += 0
                if count >= 10:
                    res = x.replace(w, "包含手机号")
                    return res
                else:
                    return x
        else:
            return x


if __name__ == "__main__":
    inputs = "FH叁伍肆柒(全部小写)您好 , 专业 保洁 大 品牌 肆 拾元 起 需要 可以 联系 薇 ↮┷▨╢⒦♭۵ⓕ❏心 zhenguoxiaopang 谢谢"
    print(inputs)
    print()
    string2py = String2py(inputs)
    pinyin, hanzi = string2py.build_string2py()
    print(pinyin, hanzi)