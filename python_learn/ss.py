d1 = {'抵达': 0, '开头语': 0, '周边': 0, '价格': 0, '目的语': 0, '结束语': 0, '交通': -1, '房间': 0, '房态': 0}
d2 = {'抵达': 0, '开头语': 0, '周边': 0, '价格': 0, '目的语': 0, '结束语': -1, '交通': -1, '房间': 0, '房态': 0}


def cmpdict(dic1, dic2):
    # dic1 数据库中现有的，　dic2　现在改变的

    assert isinstance(dic1, dict)
    assert isinstance(dic2, dict)
    out = []
    for k1, v1 in dic1.items():
        rows = {}
        v2 = dic2[k1]
        if v1 != v2:
            rows[k1] = v2
            out.append(rows)
    return out


res = cmpdict(d1, d2)
print(res)
