import re
from collections import Counter

# data_path = 'D:\\alvin_py\\business\\p201905\\0506_500\\The Merchant of Venice\\data\\Merchant of Venice_ List of Scenes.html'
data_path = input("请输入文件绝对路径：")


def readLines(filename):
    # 读文件的方法
    out = []
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            out.append(line.replace("\n", ""))
    return " ".join(out)


fw = open('Merchant of Venice_.md', 'w')
b_times = []


def read_next_content(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for content in lines:
            # 先统计次数
            b_times1 = re.findall('<\w+>', content)
            b_times2 = re.findall('<\w+ ', content)
            if b_times1 or b_times2:
                for i in b_times1:
                    b_times.append(i)
                for j in b_times2:
                    b_times.append("{}>".format(j))

            main_title = re.search('^<i>.*</i>$', content)
            if main_title:
                fw.writelines('\n')
                fw.writelines('{}\n'.format(main_title.group().replace('<i>', '*').replace('</i>', '*')))

            title_2 = re.findall("<A NAME=speech\d><b>.*</b></a>", content)
            if title_2:
                fw.writelines('\n')
                fw.writelines('**{}**\n'.format(re.findall('\w+', title_2[0])[4]))
                fw.writelines('\n')
            body = re.findall('<A NAME=\d+>.*</A><br>', content)
            if body:
                body_res = re.findall('>.*</A>', body[0])[0]
                fw.writelines('{}\n'.format(body_res.replace("</A>", "").replace(">", "")))


def get_list_scene(filename):
    p = 'Act \d, Scene \d: <a href=.*.html">.*</a><br>'
    act_num_tmp = []

    with open(filename, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for content in lines:
            # 先统计次数
            b_times1 = re.findall('<\w+>', content)
            b_times2 = re.findall('<\w+ ', content)
            if b_times1 or b_times2:
                for i in b_times1:
                    b_times.append(i)
                for j in b_times2:
                    b_times.append("{}>".format(j))

            res = re.findall(p, content)
            for i in res:
                act_num = re.findall('Act \d', i)[0]
                if act_num not in act_num_tmp:
                    act_num_tmp.append(act_num)
                    fw.writelines('\n')
                    fw.writelines("## {}\n".format(act_num))
                    fw.writelines("\n")
                scene = re.findall('Scene \d', i)[0]
                title = re.findall('>.*.</a>', i)[0].replace("</a>", "").replace(">", "")
                fw.writelines("### {} {}\n".format(scene, title))
                fw.writelines('\n')
                html_path = "./The Merchant of Venice/data/{}".format(re.findall("merchant/merchant.\d.\d.html", i)[0])
                read_next_content(html_path)


get_list_scene(data_path)

for x, y in Counter(b_times).most_common(3):
    print(x, y)

fw.close()
print("[!] Merchant of Venice_.md 已保存")