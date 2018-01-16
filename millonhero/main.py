import urllib.request
from PIL import Image
from aip import AipOcr
import urllib.request, time, _thread, urllib.parse


class Ai:
    def biggest(self, a, b, c):  # 获取出现次数最多的答案
        if a > b:
            maxnum = a
        else:
            maxnum = b
        if c > maxnum:
            maxnum = c
        return maxnum

    def __init__(self, issue, answer):  # 注意前后各两个下划线
        self.start = time.time()
        self.issue = issue
        self.answer = answer
        self.a = 0
        self.b = 0
        self.c = 0
        self.count = 0

    def gethtml(self, url):
        headers = ('User-Agent',
                   'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11')

        opener = urllib.request.build_opener()
        opener.addheaders = [headers]
        date = opener.open(url).read()

        if "zhidao.baidu.com" in url:
            str1 = date.decode('gbk').encode('utf-8').decode('utf-8')
        else:
            str1 = str(date, "utf-8")

        self.count += 1
        print(" debug : {}".format(str1))
        self.a += str1.count(self.answer[0].replace('A', ''))
        self.b += str1.count(self.answer[1].replace('B', ''))
        self.c += str1.count(self.answer[2].replace('C', ''))

    def threhtml(self, url):  # 开线程获得网页
        _thread.start_new_thread(self.gethtml, (url,))

    def search(self):
        # 可以自己添加搜索接口  self.threhtml(网址) 并在59行代码加一个数
        self.threhtml(
            "https://zhidao.baidu.com/search?lm=0&rn=10&pn=0&fr=search&ie=gbk&word=" + urllib.parse.quote(self.issue,
                                                                                                          encoding='gbk'))

        self.threhtml("http://wenwen.sogou.com/s/?w=" + urllib.parse.quote(self.issue) + "&ch=ww.header.ssda")

        self.threhtml(
            "https://iask.sina.com.cn/search?searchWord=" + urllib.parse.quote(self.issue) + "&record=1")

        self.threhtml("https://wenda.so.com/search/?q=" + urllib.parse.quote(self.issue))

        while 1:
            if (self.count == 4):  # 这里是59行代码，如果你自己增加搜索接口(4改5)
                break

        dict = {self.a: 'A', self.b: 'B', self.c: 'C'}

        listselect = [self.a, self.b, self.c]
        print('---------------------------------')
        print(' 选项    出现次数  ')
        print('  A：     ' + str(self.a))
        print('  B：     ' + str(self.b))
        print('  C：     ' + str(self.c))
        print('---------------------------------')
        print('  推荐答案：' + dict[self.biggest(self.a, self.b, self.c)])
        print('---------------------------------')
        print()

        end = time.time()
        print('搜索用时：' + str(end - self.start) + '秒')


def get_image_from_phone():
    im = Image.open(r"./screenshot.png")

    img_size = im.size
    w = im.size[0]
    h = im.size[1]
    print("xx:{}".format(img_size))

    region = im.crop((70, 200, w - 70, 1200))  # 裁剪的区域
    region.save(r"./crop_test1.png")


""" 读取图片 """


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


APP_ID = '10699696'
API_KEY = 'UL8RHjCpLrrXP47Tm2jZaTgv'
SECRET_KEY = 'Sqhs08u1c8CqY8hVaDvXr7dadScNMBoP'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

image = get_file_content("orc_test.png")
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
print("{}, {}, {}".format(answer[0], answer[1], answer[2]))

keyword = issue
ai = Ai(issue, answer)
ai.search()
