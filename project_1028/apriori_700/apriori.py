from collections import defaultdict
from itertools import combinations


def readFile(filename):
    """ 
    Function read the file 
    return a list of sets which contains the information of the transaction
    """
    originalList = list()
    file = open(filename, 'rU')
    c = 0
    limit = 1000
    for line in file:
        c = c + 1
        line = line.strip().rstrip(',')
        record = set(line.split(', '))
        originalList.append(record)
        if c > limit:
            break
    return originalList, c


def isSubList(pattern_list, sub_list):
    pattern, sublist = pattern_list[:], sub_list[:]
    while sublist:
        if sublist[-1] in pattern:
            pattern.remove(sublist[-1])
            del sublist[-1]
            if not sublist:
                return True
        else:
            return False


def generate_atom_frequence(data_sets):
    item_freq = defaultdict(int)
    for sub_list in data_sets:
        for i in sub_list:
            item_freq[i] = 1 if i not in item_freq.keys() else item_freq[i] + 1
    return item_freq


def returnConditional(num, data_sets=None):
    def subset(condition):
        for i in range(len(condition)):
            samp = list(condition[:])
            del samp[i]
            if tuple(samp) not in data_sets:
                return False
        return True

    if num == 2:
        return [i for i in combinations(data_sets, num)]
    else:
        freq = support_filter(generate_atom_frequence([list(i) for i in data_sets]), support=num - 1)
        if len(freq) < num:
            return []
        else:
            return [i for i in combinations(freq.keys(), num) if subset(i)]


def generate_frequence_dict(match_list, conditional, cond_dict=defaultdict(lambda: 0)):
    for i in conditional:
        for j in match_list:
            if isSubList(list(j), list(i)):
                cond_dict[i] += 1
    return cond_dict


def support_filter(transac, support):
    return {k: v for k, v in transac.items() if v >= support}


def apriori(transactions, minSupport):
    freq_sets = support_filter(generate_atom_frequence(transactions), support=minSupport)
    k = 2
    while True:
        print(k)
        condition = returnConditional(num=k, data_sets=freq_sets)
        frequence_dict = generate_frequence_dict(transactions, condition)
        freq_sets = support_filter(frequence_dict, support=minSupport)
        k += 1
        if condition:
            print(condition, )
        if not condition:
            break
        if not freq_sets:
            break

    fw = open("apriori_results.txt", 'w', encoding='utf-8')
    for k, v in freq_sets.items():
        print(k, v)
        res = '{},{}'.format(k, v)
        fw.writelines(res + '\n')


globOriginalList, globNumberOfTransactions = readFile('adult.data.txt')
minSupport = 3
apriori(globOriginalList, 3)
