import jieba
import re


def replaces(inputs):
    return re.sub("[+“”！，,。？、&*\ ]", "", inputs)


groups = [
    '世界说大很大,说小很小，',
    '大到走了那么久, 还没跟对的人相遇',
    ' 小到围着喜欢的人绕一圈,就看到了全世界。',
    ' 我与世界只差一个你。'
]


def getTotalwords(groups):
    totoal_words = {}
    k = 1
    totoal_words_cut = []
    for one in groups:
        cuts = jieba.lcut(replaces(one.replace(" ", "")))
        totoal_words_cut.append(cuts)
        for x in cuts:
            totoal_words[x] = k
            k += 1
    return totoal_words, totoal_words_cut


totoal_words, totoal_words_cut = getTotalwords(groups=groups)


def get_vec(inputs):
    rows = []
    for one in inputs:
        n = totoal_words[one]
        rows.append(n)
    return rows

