from collections import Counter
import pandas as pd


x1 = []
with open("D:\\alvin_py\\for_Order\\fd.csv", 'r', encoding='gbk') as fr:
    lines = fr.readlines()
    for line in lines:
        x1.append(line.replace('\n', ''))

print(len(x1))
print(len(list(set(x1))))

save = []
for x, y in Counter(x1).most_common(30):
    print(x, y)
    save.append("{},{}".format(x,y))

print(len(save))


# df = pd.DataFrame(save)
# df.to_csv("重复.csv", index=None)

# 283 1674