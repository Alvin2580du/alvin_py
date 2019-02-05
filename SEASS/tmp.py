
num = 0

with open("./clean_data/train.source", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        num += 1

print(num)