from pyduyp.datasources.xiaohua import add, getone, update

import datetime

from pyduyp.utils.fileops import curlmd5


def add_xiaohua2db():
    xiaohua_path = "./datasets/xiaohua.txt"
    with open(xiaohua_path, 'r', encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            newline = line.replace("\n", "").lstrip().rstrip()
            inputs = {"name": newline, "namemd5": curlmd5(newline), "status": 0, "createtime": datetime.datetime.now()}
            ret = add(inputs_dict=inputs)
            print(ret)


def dbtest():
    search = getone()
    print(search['id'])
    updateinputs = {"name": search['name'], "status": 1, "createtime": datetime.datetime.now()}
    ret = update(idstr=search['id'], inputs_dict=updateinputs)
    print(ret)

if __name__ == "__main__":
    dbtest()
