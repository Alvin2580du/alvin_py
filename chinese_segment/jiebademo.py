import jieba
import jieba.posseg
import jieba.analyse
import re


def newcut(inputs):
    if isinstance(inputs, str):
        _msg_cut = jieba.lcut(inputs)
        print("|".join(_msg_cut))
        print("="*20)
        # 这里可以用正则匹配出文本出现价格的词，
        # 因为类似100元这样的词，jieba是分不出来的，
        # 但是你不能把类似这样的词加到词典里面去吧，
        # 否则词典会变的很大。
        p = re.compile("[0-9]+?[元|块]").findall(inputs)
        if p:
            for price in p:
                _msg_cut.append(price)
        if len(_msg_cut) > 0:
            return "|".join(_msg_cut)
        else:
            return inputs
    else:
        return inputs

inputs = "今天是情人节，娜娜给我发了52元的红包!"
print(newcut(inputs))

"""

今天|是|情人节|，|娜娜|给我发|了|52|元|的|红包|!
===========================================
今天|是|情人节|，|娜娜|给我发|了|52|元|的|红包|!|52元
"""