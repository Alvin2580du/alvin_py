import urllib.request
from aip import AipOcr
import urllib.request
import time
import os
import re
import cv2
import urllib.parse

"""
来源: https://github.com/wuditken/MillionHeroes.git

"""

start = time.time()

os.system("adb shell /system/bin/screencap -p /sdcard/screenshot.png")
os.system("adb pull /sdcard/screenshot.png screenshot.png")

img = cv2.imread("screenshot.png")
crop_img = img[200:1200, 90:1000]
cv2.imwrite("crop_test1.png", crop_img)


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def search_answer(issue, answer):
    url1 = "https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=" + urllib.parse.quote(issue,
                                                                                                         encoding='gbk')
    url2 = "https://iask.sina.com.cn/search?searchWord=" + urllib.parse.quote(issue)

    count = 0
    res = ""

    url = url2
    headers = ('User-Agent',
               'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 '
               'Safari/537.11')

    opener = urllib.request.build_opener()
    opener.addheaders = [headers]
    date = opener.open(url).read()

    if "zhidao.baidu.com" in url:
        str1 = date.decode('gbk').encode('utf-8').decode('utf-8')
        for x in str1:
            if is_chinese(x):
                res += x
    else:
        str1 = str(date, "utf-8")
        for x in str1:
            if is_chinese(x):
                res += x

    count += 1
    A = answer[0].replace('A', '')
    B = answer[1].replace('B', '')
    C = answer[2].replace('C', '')
    a = len(re.compile(A).findall(res))
    b = len(re.compile(B).findall(res))
    c = len(re.compile(C).findall(res))

    print("搜索结果：\n {}".format(res))
    dicts = {a: 'A', b: 'B', c: 'C'}
    print('---------------------------------')
    print(' 选项    出现次数  ')
    print("A : {}  B: {},  C: {}".format(a, b, c))

    print('---------------------------------')
    print('  推荐答案：{} '.format(dicts[max([a, b, c])]))
    print('---------------------------------')
    print()

    end = time.time()
    print('搜索用时：' + str(end - start) + '秒')


APP_ID = '10699696'
API_KEY = 'UL8RHjCpLrrXP47Tm2jZaTgv'
SECRET_KEY = 'Sqhs08u1c8CqY8hVaDvXr7dadScNMBoP'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

image = get_file_content("crop_test1.png")

respon = client.basicGeneral(image)
titles = respon['words_result']
issue = ''
answer = ['', '', '', '', '', '']
countone = 0
answercount = 0
for title in titles:
    countone += 1
    if countone >= len(titles) - 2:
        answer[answercount] = title['words']
        answercount += 1
    else:
        issue = issue + title['words']

tissue = issue[1:2]
if str.isdigit(tissue):
    issue = issue[3:]
else:
    issue = issue[2:]

print("问题: {}".format(issue))
print("- " * 20)
print("A: {} \n B: {} \n C: {}".format(answer[0], answer[1], answer[2]))

search_answer(issue, answer)
