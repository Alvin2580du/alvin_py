import docx
import pandas as pd
import re
from collections import OrderedDict

doc_file = docx.Document('12967518_老上海行名辞典  英汉对照  1880-1941_p651.docx')
all_ = []

for paragraph in doc_file.paragraphs:
    # 按照段落把所有句子添加到all_列表里面
    res = paragraph.text.split("；")
    if len(res) == 2:
        all_.append(res[0])
    else:
        all_.append(paragraph.text)

save = []
names = []
for i in range(len(all_)):
    if i+2 < len(all_):
        strings = "{} {} {}".format(all_[i], all_[i+1], all_[i+2]).replace('\n', "")  # 每3个句子组成一个长句子，用来匹配
        # 加粗的内容
        name = re.compile('[A-Z][a-zA-Z\.\ \&\-\,\'\^\*\、\(\)\/0-9\，]{0,35}[\u4e00-\u9fa5]').search(strings)
        # 汉字
        hanzi = re.compile('[\u4e00-\u9fa5]+ [\u4e00-\u9fa5]*').search(strings)
        if hanzi:
            # 地址
            dizhi = re.compile('[\d+a-z\.\ \&\-\,\'\、\^\*\,A-Z\u4e00-\u9fa5]*').search(strings.split(hanzi.group())[-1])
            # 日期
            date = re.compile('\([0-9]*\-*[0-9]*[A-Z0-9]*\)').search(strings)
            if name and hanzi and dizhi and date:
                dates = date.group().replace(")", "").replace("(", "")
                try:
                    # 分割开始时间和结束时间
                    start = dates.split('-')[0]
                    end = dates.split('-')[1]
                except:
                    start = dates.split('-')[0]
                    end = ''

                # 对name做处理，把不要的信息替换掉。
                new_name = name.group().replace(name.group()[-1], "").replace('Rd.', '').replace(start, '').replace(end, '').replace("(", '').replace(")", '')
                rows = OrderedDict({"name": new_name,
                                    'hanzi': hanzi.group(),
                                    "dizhi": dizhi.group().split('Rd.')[0] + 'Rd.',
                                    "start": start,
                                    'end': end})
                if hanzi.group() not in names:
                    names.append(hanzi.group())  # 去重处理
                    save.append(rows)
                    print(rows)

df = pd.DataFrame(save)
df.to_excel("结果.xlsx", index=None)
print(df.shape)
