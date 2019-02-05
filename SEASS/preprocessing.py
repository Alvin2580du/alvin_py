import os
import jieba
from tqdm import trange
import json

sw = []
with open('./data/stopwords.csv', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        sw.append(line.replace("\n", ""))

start_tok = "<s>"
end_tok = "</s>"
unk_tok = "<unk>"
pad_tok = "<pad>"


def build_vocab(file, vocab_file, min_count=5):
    print("Building vocab with min_count=%d..." % min_count)
    freq = {}
    fin = open(file, "r", encoding="utf8")
    for _, line in enumerate(fin):
        for word in line.strip().split():
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 0
    fin.close()
    print('Number of all words: %d' % len(freq))

    vocab = {start_tok: 0, end_tok: 1, unk_tok: 2, pad_tok: 3}
    if unk_tok in freq:
        freq.pop(unk_tok)
    for word in freq:
        if freq[word] > min_count:
            vocab[word] = len(vocab)
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab) / len(freq) * 100))

    json.dump(vocab, open(vocab_file, 'w'))
    return freq


def extract_LCSTS(origin, is_partI=False):
    if is_partI:
        tmp = 0
    else:
        tmp = 1

    summaries = []
    articles = []

    with open(origin, encoding='utf-8') as f_origin:
        lines = f_origin.read().splitlines()
        for i in trange(0, len(lines), 8 + tmp):
            if not is_partI:
                score_line = lines[i + 1].strip()
                if int(score_line[13]) < 3:
                    continue

            summaries.append(lines[i + 2 + tmp].strip())
            articles.append(lines[i + 5 + tmp].strip())

    return articles, summaries


def save_data(x, y, output_dir, prefix):
    with open("{}/{}.target".format(output_dir, prefix), 'w', encoding='utf-8') as tgt_output, open(
            "{}/{}.source".format(output_dir, prefix), 'w', encoding='utf-8') as src_output:
        tgt_output.write('\n'.join(y))
        src_output.write('\n'.join(x))


def main():
    # Arguments
    PART_I_data = './data/LCSTS2.0/PART_I.txt'
    PART_III_data = './data/LCSTS2.0/PART_III.txt'
    output_dir = './clean_data/'

    # Extract data
    partI_x, partI_y = extract_LCSTS(PART_I_data, is_partI=True)
    partIII_x, partIII_y = extract_LCSTS(PART_III_data)

    # Remove overlapping data
    overlap_cnt = 0

    clean_partI_x = []
    clean_partI_y = []

    for idx in range(len(partI_x)):
        if partI_y[idx] in partIII_y:
            overlap_cnt += 1
        else:
            clean_partI_x.append(partI_x[idx])
            clean_partI_y.append(partI_y[idx])

    dirname = os.path.dirname(output_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    save_data(partIII_x, partIII_y, output_dir, 'test')
    save_data(clean_partI_x, clean_partI_y, output_dir, 'train')

    print("Remove {} pairs".format(overlap_cnt))


def cut_data():
    fw = open("./clean_data/test.target.cut", 'w', encoding='utf-8')
    num = 1
    limit = 10000
    with open('./clean_data/test.target', 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if line:
                num += 1
                line_cut = jieba.lcut(line)
                line_w = [i for i in line_cut if i not in sw]
                fw.writelines(" ".join(line_w))
                if num > limit:
                    break
            else:
                break


build_vocab(file='./clean_data/train.source.cut', vocab_file='vocab_zh.json')
