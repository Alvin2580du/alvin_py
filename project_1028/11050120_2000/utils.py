
def max_in_dict(d):
    # 定义一个求字典中最大值的函数
    key, value = 0, 0
    for i, j in d.items():
        if j > value:
            key, value = i, j
    return key, value


def viterbi_(nodes, trans):
    # viterbi算法
    paths = nodes[0]  # 初始化起始路径
    for l in range(1, len(nodes)):  # 遍历后面的节点
        paths_old, paths = paths, {}
        for n, ns in nodes[l].items():  # 当前时刻的所有节点
            max_path, max_score = '', -1e10
            for p, ps in paths_old.items():  # 截止至前一时刻的最优路径集合
                score = ns + ps + trans[p[-1] + n]  # 计算新分数
                if score > max_score:  # 如果新分数大于已有的最大分
                    max_path, max_score = p + n, score  # 更新路径
            paths[max_path] = max_score  # 储存到当前时刻所有节点的最优路径
    key, value = max_in_dict(paths)
    return key, value


def get_sents(datasets):
    # 数据读取的函数
    sents = []
    tmp = []
    words = []
    with open(datasets, 'r', encoding='utf-8') as fr:
        while True:
            lines = fr.readline()
            if lines:
                if len(lines) > 2:
                    lines_sp = lines.split(" ")
                    w, label = lines_sp[0], lines_sp[1].replace("\n", "")  # 取出单字和标签
                    tmp.append((w, label))
                    if w not in words:
                        # 这里把所有的单词做个统计，
                        words.append(w)
                else:
                    # 如果一句话结束了， 会有一个空行
                    sents.append(tmp)
                    tmp = []
            else:
                break
    return sents, words