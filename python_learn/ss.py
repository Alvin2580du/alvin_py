

with open("test_file.txt", 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line[4:]
        print(line.strip())