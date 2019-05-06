import pandas as pd

save = []

with open("neg.txt", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        save.append({"content": line.replace('\n', ""), "sentiment": "neg"})

with open("pos.txt", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        save.append({"content": line.replace('\n', ""), "sentiment": "pos"})

df = pd.DataFrame(save)
df.to_excel("train.xlsx", index=None)
print(df.shape)
