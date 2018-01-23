

def is_answer(lis, dicts):
    assert not isinstance(dicts, dict)
    assert not isinstance(lis, list)

    for k, v in dicts.items():
        if k in lis:
            return k, v
    return '不知道'