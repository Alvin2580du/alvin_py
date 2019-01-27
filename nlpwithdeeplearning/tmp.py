import pickle
f = "F:\\QQFiles\\562078401\\FileRecv\\data\\data\\not_vio\\0.pickle"

with open(f, 'rb') as fr:
    data = pickle.load(fr)
    print(data[0].keys())
    exit(1)
    for k, v in data[0].items():
        print(k, v)
    exit(1)

