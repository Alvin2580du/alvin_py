import pandas as pd

G_labels = ['无', '0-2000', '2000-3999', '4000-5999', '6000-7999', '8000-9999', '1万以上']
AA_labels = ['包含3项', '包含其中的一或两项', '完全不包含']


def G_preprocess(inputs):
    try:
        if '万' in inputs:
            try:
                tmp_ = int(int(inputs.split("-")[1].replace("元/年", "").replace("万", "0000")) / 12)
            except:
                tmp_ = int(20000 / 12)
        elif '元/年' in inputs:
            tmp_ = int(int(inputs.split("-")[1].replace("元/年", "")) / 12)
        else:
            try:
                tmp_ = int(inputs.split("-")[1].replace("元/月", "").replace("以下元/月", ""))
            except:
                tmp_ = '无'
        return tmp_
    except:
        return '无'


def label_G(inputs):
    if isinstance(inputs, int):
        if 0 <= inputs <= 2000:
            return G_labels[1]
        if 2000 <= inputs <= 3999:
            return G_labels[2]
        elif 4000 <= inputs <= 5999:
            return G_labels[3]
        elif 6000 <= inputs <= 7999:
            return G_labels[4]
        elif 8000 <= inputs <= 9999:
            return G_labels[5]
        elif inputs >= 10000:
            return G_labels[6]
        else:
            return inputs
    else:
        return G_labels[0]


"""
技能/语言
思就是说包含Photoshop、Corel DRAW,AI这三个技能的分成一类，包含其中的一或两项的分成一类，完全不包含的分成一类
"""


def label_AA(inputs):
    if inputs == 'PS,AI.DW,(精通)' or inputs == 'Photoshop/CDR(良好)':
        return AA_labels[1]
    if inputs == '英语(熟练)  Word/Excel/ PPT/Photoshop(熟练)  bootstrap(良好)  Jquery(熟练)  HTML/CSS/JavaScript(精通)':
        return AA_labels[1]

    tmp_ = []

    for item in str(inputs).lower().split('  '):
        for item2 in item.split("、"):
            item3 = item2.replace("熟练", "").replace("一般", "").replace("良好", "").replace(")", "").replace("(", "")
            tmp_.append(item3.replace("精通", "").replace(" adobe photoshop", 'photoshop'))

    print(tmp_)

    label1 = ['Photoshop', 'Corel DRAW', 'AI']
    num = 0
    for i in label1:
        for j in tmp_:
            if i.lower() == j:
                num += 1

    if num == 3:
        return AA_labels[0]
    elif num == 2 or num == 1:
        return AA_labels[1]
    else:
        return AA_labels[2]


data = pd.read_excel("data1.xlsx")
print("开始打标签...")
data['G_preprocess'] = data['期望薪资'].apply(G_preprocess)
data['G_label'] = data['G_preprocess'].apply(label_G)
data['AA_label'] = data['技能/语言'].apply(label_AA)
del data['G_preprocess']
data.to_excel("results.xlsx", index=None)
print('保存成功！')
