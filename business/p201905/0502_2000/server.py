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
import random
import matplotlib.pyplot as plt
from tornado.web import Application
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    'all': "全部", 'twelve': "12级", 'thirteen': "13级", 'fourteen': "14级",
    'fifteen': "15级", 'sixteen': "16级", 'seveteen': "17级",
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
    '1': "一级",
    '2': "二级",
    '3': "三级",
    '4': "四级",
    '5': "五级",
    '6': "六级",
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


shi_pr = {'内蒙古自治区': '呼和浩特', '新疆维吾尔自治区': '乌鲁木齐', '西藏': '拉萨', '宁夏回族自治区': '银川', '广西壮族自治区': '南宁',
          '香港特别行政区': '香港', '澳门特别行政区': '澳门', '黑龙江省': '哈尔滨', '吉林省': '长春', '辽宁省': '沈阳',
          '河北省': '石家庄', '山西省': '太原', '青海省': '西宁', '.山东省': '济南', '河南省': '郑州', '江苏省': '南京',
          '安徽省': '合肥', '浙江省': '杭州', '福建省': '福州', '江西省': '南昌', '湖南省': '长沙', '湖北省': '武汉',
          '广东省': '广州', '台湾省': '台北', '海南省': '海口', '甘肃省': '兰州', '陕西省': '西安', '四川省': '成都',
          '贵州省': '贵阳', '云南省': '昆明', '北京市': '北京', '天津市': '天津', '重庆市': '重庆', '上海市': '上海', '深圳市': '深圳'}

shi_pr_re = {v: k for k, v in shi_pr.items()}


# 全校各省学生二级的通过率，用中国地图散点图表示出来
def get_province():
    rows_lat = {}
    rows_lon = {}
    with open('weidu.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line_sp = line.split(',')
            rows_lat[shi_pr_re[line_sp[0].replace('市', '')]] = line_sp[1].replace("\n", "")
            rows_lon[shi_pr_re[line_sp[0].replace('市', '')]] = line_sp[2].replace("\n", "")

    print(rows_lon)
    print(rows_lat)

    values = []
    names = []
    save = []
    for x, y in data.groupby(by='province'):
        try:
            for x1, y1 in y.groupby(by='dengji'):
                if x1 == '二级':
                    lelve = [i for i in y1['chengji'] if i > 60]
                    rate = len(lelve) / y1.shape[0]
                    rows = {}
                    rows['name'] = x
                    rows['values'] = "{:0.2f}".format(rate)
                    rows['lat'] = rows_lat[x.replace("\n", "")]
                    rows['lon'] = rows_lon[x.replace("\n", "")]
                    names.append(x)
                    values.append("{:0.2f}".format(rate))
                    save.append(rows)
        except:
            print(x)
            continue

    df = pd.DataFrame(save)
    df.to_excel("province.xlsx", index=None)

    plt.figure(figsize=(20, 10))
    plt.bar(names, values, width=0.5)
    plt.xticks(rotation=36)
    plt.savefig("./static/images/各省二级通过率对比.png")
    print("save success ")


def plot_province():
    import numpy as np
    import matplotlib

    matplotlib.rcParams['toolbar'] = 'None'
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import pandas as pd

    posi = pd.read_excel("province.xlsx")

    lat = posi["lat"].values.tolist()  # 获取维度之维度值
    lon = posi["lon"].values.tolist()  # 获取经度值
    values = posi["values"].values.tolist()  # 获取经度值
    size = (values / np.max(values)) * 100

    map = Basemap(projection='stere',
                  lat_0=35, lon_0=110,
                  llcrnrlon=82.33,
                  llcrnrlat=3.01,
                  urcrnrlon=138.16,
                  urcrnrlat=53.123, resolution='l', area_thresh=10000, rsphere=6371200.)

    map.drawcoastlines()
    map.drawcountries()
    map.readshapefile("gadm36_CHN_shp/gadm36_CHN_0", 'states', drawbounds=True)
    map.drawmapboundary()

    parallels = np.arange(0., 90, 10.)
    map.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)  # 绘制纬线

    meridians = np.arange(80., 140., 10.)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)  # 绘制经线

    x, y = map(lon, lat)

    map.scatter(x, y, edgecolors='r', facecolors='r', marker='*', s=size)
    plt.savefig("scatter.png")
    plt.show()


def scatter_plot():
    rows = {}

    for x, y in data.groupby(by='dengji'):
        chengji = y['chengji'].values.tolist()
        rows[x] = random.sample(chengji, 200)

    plt.figure()
    plt.scatter(list(rows['二级']), list(rows['三级']))
    plt.title("二级和三级分数散点图")
    plt.savefig("./static/images/23.png")

    plt.close()

    plt.figure()
    plt.scatter(list(rows['三级']), list(rows['四级']))
    plt.title("三级和四级分数散点图")
    plt.savefig("./static/images/34.png")
    plt.close()


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
        print("init GradeHandler handler")

    def get(self):
        # 拿到参数后返回
        self.render('main.html')

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

        data = {"scores": scores_, "rate": rate_}
        print(data)
        self.write(json.dumps(data))


class ScatterHandler(tornado.web.RequestHandler):
    def post(self):
        self.render('scatter.html')


class VisiHandler(tornado.web.RequestHandler):
    def post(self):
        self.render('scatter.html')


class ProvinceHandler(tornado.web.RequestHandler):
    def post(self):
        self.render('province.html')


if __name__ == "__main__":

    method = 'get_province'  # server

    if method == 'scatter_plot':
        scatter_plot()
    elif method == 'get_province':
        get_province()
    else:
        tornado.options.parse_command_line()
        handlers = [
            (r"/", IndexHandler),
            (r"/user", LoginHandler),
            (r"/grade", GradeHandler),
            (r"/scatter", ScatterHandler),
            (r"/province", ProvinceHandler),

        ]
        app = Application(handlers,
                          template_path=os.path.join(os.path.dirname(__file__), "templates"),
                          static_path=os.path.join(os.path.dirname(__file__), "static"),
                          debug=True
                          )
        http_server = tornado.httpserver.HTTPServer(app)
        http_server.listen(options.port)
        print("http_server 已启动")
        tornado.ioloop.IOLoop.instance().start()
