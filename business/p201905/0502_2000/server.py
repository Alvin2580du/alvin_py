import os.path
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import traceback
import json
import logging
import pandas as pd

ports = 8994
define("port", default=ports, help="run on the given port", type=int)

data = pd.read_excel("全部需要数据.xlsx")
xueqi_dict = {'all': "全部", "chunji": "春季", 'qiuji': "秋季"}

xueyuan_dict = {
    'all': "全部", "tmgc": "土木工程学院", 'wgy': "外国语学院", 'wxy': "文学院",
    'xny': "新能源学院", 'jdgc': "机电工程学院", 'clkxygc': "材料科学与工程学院",
    'fxy': "法学院", 'lxy': "理学院", 'smkx': "生命科学与工程学院",
    'dqxx': "电气工程与信息工程学院", 'syhg': "石油化工学院",
    'jjgl': "经济管理学院", 'nydl': "能源与动力工程学院",
    'jsjtx': "计算机与通信学院", 'sjys': "设计艺术学院", 'rjxy': "软件学院",
}
nianji_dict = {
    'all': "全部", 'twelve': "12", 'thirteen': "13", 'fourteen': "14",
    'fifteen': "15", 'sixteen': "16", 'seveteen': "17",
}

yuzhong_dict = {
    'all': "全部", "access": "Access数据库程序设计",
    "cplus": "C++语言程序设计",
    "cyy": "C语言程序设计",
    "java": "Java语言程序设计",
    "msoffice": "MS Office高级应用",
    "vbyy": "VB语言程序设计",
    "web": "Web程序设计",
    "xxaq": "信息安全技术",
    "qrs": "嵌入式系统开发技术",
    "sjk": "数据库技术",
    "wljs": "网络技术",
    "jsjms": "计算机基础及MS Office应用",
    "jsjps": "计算机基础及Photoshop应用",
    "jsjwps": "计算机基础及WPS Office应用",
}

grade_dict = {
    'all': "全部",
    '1': "一",
    '2': "二",
    '3': "三",
    '4': "四",

}


# 各个学院各个等级
def get_1(xq, xy, grade):
    for x, y in data.groupby(by='xueqi'):
        for x1, y1 in y.groupby(by='xueyuan'):
            for x2, y2 in y1.groupby(by='dengji'):
                if x == xq:
                    if x1 == xy:
                        if grade == x2:
                            scores = y2['chengji'].mean()
                            lelve = [i for i in y2['chengji'] if i > 60]
                            rate = len(lelve) / y2.shape[0]
                            print('{:0.3f}'.format(scores), "{:0.3f}".format(rate))
                            print("- * -" * 10)
                            return '{:0.3f}'.format(scores), "{:0.3f}".format(rate)
    print("{},{},{}无考试学生".format(xq, xy, grade))
    return 11111, 99999


# 各个学院各个等级语种
def get_2(xq, xy, grade, yuzhong):
    for x, y in data.groupby(by='xueqi'):
        for x1, y1 in y.groupby(by='xueyuan'):
            for x2, y2 in y1.groupby(by='dengji'):
                for x3, y3 in y2.groupby(by='yuzhong'):
                    if x == xq:
                        if x1 == xy:
                            if x2 == grade:
                                if x3 == yuzhong:
                                    scores = y3['chengji'].mean()
                                    lelve = [i for i in y3['chengji'] if i > 60]
                                    rate = len(lelve) / y3.shape[0]
                                    print('{:0.3f}'.format(scores), "{:0.3f}".format(rate))
                                    print("- * -" * 10)
                                    return '{:0.3f}'.format(scores), "{:0.3f}".format(rate)
    print("{},{},{},{}无考试学生".format(xq, xy, grade, yuzhong))
    return 11111, 99999


# 各个年级各个等级
def get_3(xq, nj, grade):
    for x, y in data.groupby(by='xueqi'):
        for x1, y1 in y.groupby(by='nianji'):
            for x2, y2 in y1.groupby(by='dengji'):
                if x == xq and x1 == nj and x2 == grade:
                    scores = y2['chengji'].mean()
                    lelve = [i for i in y2['chengji'] if i > 60]
                    rate = len(lelve) / y2.shape[0]
                    print('{:0.3f}'.format(scores), "{:0.3f}".format(rate))
                    print("- * -" * 10)
                    return '{:0.3f}'.format(scores), "{:0.3f}".format(rate)
    print("{},{},{}无考试学生".format(xq, nj, grade))
    return 11111, 99999


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("login.html")


class LoginHandler(tornado.web.RequestHandler):
    def post(self):
        userName = self.get_argument('name')
        password = self.get_argument('password')
        if userName == 'admin' and password == '123456':
            self.render("main.html")
        else:
            self.render('08login.html', error='登陆失败')

    def get(self):
        # 拿到参数后返回
        dict = {}
        try:
            res = json.dumps(dict)
            self.finish(res)
        except Exception as e:
            logging.warning(e)
            traceback.print_exc()


class GradeHandler(tornado.web.RequestHandler):

    def initialize(self):
        print("init KBRecallHandler handler")

    def get(self):
        # 拿到参数后返回
        dict = {}
        try:
            res = json.dumps(dict)
            print("res:{}".format(res))
            self.finish(res)
        except Exception as e:
            logging.warning(e)
            traceback.print_exc()

    def post(self):
        print('post = * =' * 10)
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        xq = xueqi_dict[self.get_argument('select-xueqi', None)]
        xy = xueyuan_dict[self.get_argument("select-xueyuan", None)]
        nianji = nianji_dict[self.get_argument("select-nianji", None)]
        grade = grade_dict[self.get_argument("select-grade", None)]
        yuzhong = yuzhong_dict[self.get_argument("select-yuzhong", None)]
        print("学期：{}, 学院：{}， 年级：{}， 等级：{}, 语种：{}".format(xq, xy, nianji, grade, yuzhong))
        # 需要分析各个学院各个等级（如二级，三级，四级）的平均分，通过率
        if xq and xy and grade and nianji == '全部' and yuzhong == '全部':
            print("第1个问题")
            scores_, rate_ = get_1(xq, xy, grade)
        # # 各个学院各个等级语种
        elif xq and xy and grade and yuzhong and nianji == '全部':
            print("第2个问题")

            scores_, rate_ = get_2(xq, xy, grade, yuzhong)
        # # 各个年级各个等级
        elif xq and nianji and grade and xy == '全部' and yuzhong == '全部':
            print("第3个问题")
            scores_, rate_ = get_3(xq, nianji, grade)
        else:
            print("没有定义的问题")
            scores_, rate_ = 11111, 99999

        pd = {"学期": xq, "学院": xy, "年级": nianji, "语种": grade, "等级": yuzhong}
        data = {"inputs": pd, "scores": scores_, "rate": rate_}
        print(data)
        self.write(json.dumps(data))


handlers = [
    (r"/", IndexHandler),
    (r"/user", LoginHandler),
    (r"/grade", GradeHandler)
]

if __name__ == "__main__":
    template_path = os.path.join(os.path.dirname(__file__), "template")
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers, template_path)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    print("http_server 已启动")
    tornado.ioloop.IOLoop.instance().start()
