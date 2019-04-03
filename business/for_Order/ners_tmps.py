import pandas as pd

cidian_score = pd.read_csv("cidian_score.csv")
cidian_score_dict = {}

for x, y in cidian_score.iterrows():
    name = y['name']
    score = y['score']
    cidian_score_dict[name] = score


def juzi2list(juzi):
    sentences = juzi.lower().split('.')
    save = []
    for sen in sentences:
        out = []
        for x in sen.split():
            out.append(x.replace(".", ""))
        save.append(out)
    return save

# The Great Wall was absolutely stunning
def get_eight_words(l, x):
    if x in l:
        index = l.index(x)
        if index < 8:
            left = l[:index + 1]
            right = l[index: index + 8]
            all_words = left + right
            return " ".join(all_words)

        else:
            left = l[index - 8:index + 1]
            right = l[index: index + 8]
            all_words = left + right
            return " ".join(all_words)
    else:
        return 0


def make_dict(find_socre):
    keys_words_dict = {"place": 0, 'trip': 0, 'entrance': 0, 'step': 0, 'water': 0, 'view': 0, 'park': 0, 'crowd': 0,
                       'food': 0, 'people': 0, 'tower': 0, 'mountain': 0, 'experience': 0, 'wall': 0, 'ticket': 0,
                       'building': 0, 'history': 0, 'cable car': 0, 'bus': 0, 'guide': 0, 'hotel': 0}

    for x in find_socre:
        keys_words_dict[list(x.keys())[0]] = list(x.values())[0]
    return keys_words_dict

keys_words = ['place', 'trip', 'entrance', 'step', 'water', 'view', 'park', 'crowd', 'food', 'people', 'tower',
              'mountain', 'experience', 'wall', 'ticket', 'building', 'history', 'cable car', 'bus', 'guide', 'hotel']


def get_sentence_score(sentence):
    res = []
    for x in keys_words:
        max_score = get_eight_words(sentence, x)
        try:
            for jj in list(cidian_score_dict.keys()):

                if jj in max_score:
                    score = cidian_score_dict[jj]
                    rows = {x: score}
                    res.append(rows)
                else:
                    rows = {x: 4}
                    res.append(rows)
        except Exception as e:
            continue
    return res


def get_biaoti_score(line):
    all_sentences = juzi2list(line)
    save = []

    for sentence in all_sentences:
        res = get_sentence_score(sentence)
        for x in res:
            if x not in save:
                save.append(x)
    return save


def build_six():
    with open("xuqiu_2.csv", 'r', encoding='utf-8') as fr:
        k = 0
        limit = 1000000
        save_df = []
        while True:
            line = fr.readline()
            if line:
                k += 1
                if k == 1:
                    continue
                get_biaoti_score_res = get_biaoti_score(line=line)
                get_dict = make_dict(get_biaoti_score_res)
                get_dict['标题'] = line.replace("\n", "")
                save_df.append(get_dict)
                if k > limit:
                    break
                if k % 100 == 1:
                    print(k)
            else:
                break
        df = pd.DataFrame(save_df)
        df.to_csv("评论内容——分数.csv", index=None, encoding='utf-8')

build_six()
