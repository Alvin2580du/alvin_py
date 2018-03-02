import re
import opencc


def removespecialchar(str):
    # log.debug("removespecialchar: {}".format(str))
    if str is None or len(str) == 0:
        return ''
    return str.strip().replace("\r\n", '').replace("\n", '')


def is_question(inputs):
    """
    ['岂', '何尝', '究竟', '几', '怎么着', '多少 怎么', '怎样', '居然', '?', '难怪', '吗', '怎么样', '难道', '怎么', '吧', '如何',
                   '几时', '怎', '为什么', '竟然', '多少', '何必', '反倒', '什么', '啊', '怎的', '几时', '何', '什么', '简直', '呢',
                   '谁', '哪里', '哪儿']

    regex = "[\何\怎样\吧\如何\岂\为什么\反倒\多少\怎么\啊\怎的\怎么着\什么\难道\呢\怎么样\竟然\谁\究竟\何必\?\吗\居然\难怪\哪儿\几时\何尝\怎么\几\多少\简直\怎\哪里]"

    """
    outputs = False
    w = None
    question_list_1 = ['岂', '何尝', '究竟', '几', '怎么着', '多少', '怎么', '怎样', '居然', '?', '难怪', '吗', '怎么样', '难道', '怎么', '吧',
                       '如何', '几时', '怎', '为什么', '竟然', '何必', '反倒', '什么', '啊', '怎的', '几时', '何', '什么', '简直', '呢',
                       '谁', '哪里', '哪儿', '？']
    question_list_2 = ['麻烦发下', '多远', '嘛', '请问', '能不能', '方便不', '么', '没有', '多钱', '呗',
                       '咋办', '有啥', '行不行', '不成功', '多长时间', '是否', '哇', '哪站', '多久', '近不近', '哪个站',
                       '还是', '请告知', '嗎', '免费不', '收费不', '可以', '方不方便', '有房没', '有房吗', '不', "远不远"]

    question_list = question_list_1 + question_list_2
    for i in question_list:
        if i in inputs:
            outputs, w = True, i

    if w is None:
        return outputs
    else:
        return outputs, w


def dict_process(inputs, k):
    # 求字典排除一个元素之后剩余值的方法
    """
    commond_list = {"距离": "到去离来近远怎么走几路公交房子家地铁",
                    "时间": "时间多久过来要",
                    "价钱": "收费钱价格贵便宜"}
    res = dict_process(inputs=commond_list, k='距离')
    #　时间多久过来要收费钱价格贵便宜
    """
    data = dict(inputs)
    out = []
    for key, value in data.items():
        if key == k:
            continue
        else:
            out.append(value)
    return "".join(out)




cc = opencc.OpenCC('t2s')
def linesplit_bysymbol(line):
    # 先按句子切割，然后去标点，然后繁简体转换
    out = []
    juzi = r"[\】\【\：\，\。\?\？\)\(\,\.\(\『\』\<\>\、\；\．\[\]\（\）\〔\〕]"
    p = r"[\^\$\]\］\［\/\.\’\~\#\￥\#\&\*\%\”\“\]\[\&\×\@\]\"]"
    linesplit = re.split(juzi, line)
    for x in linesplit:
        res = re.sub(p, "", x)
        resnew = ""
        for i in res:
            t2s = cc.convert(i)
            resnew += t2s
        out.append(resnew.replace("\n", ""))
    return out
