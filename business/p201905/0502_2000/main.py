"""
1.系统需要一个登录页面，图形用户界面。
2.需要分析各个学院各个等级（如二级，三级，四级）的平均分，通过率；
各个学院各个等级语种（如二级c语言/office...三级数据库/计算机网络技术...）的平均分以及通过率；
各个年级各个等级的平均分和通过率；
2018春季学期和秋季学期对比分析。
3，将数据用图表，散点图可视化出来
4.全校各省学生二级的通过率，用中国地图散点图表示出来


# 数据格式

姓名 性别 省份 学院 学号 学期 成绩 语种 等级

"""
import pandas as pd
import os
from collections import OrderedDict

data_path_3 = '2018年全国等级考试成绩汇总及学籍信息\\2018年3月NCRE_lut_成绩'
data_path_9 = '2018年全国等级考试成绩汇总及学籍信息\\2018年9月NCRE_lut_成绩'
data_path_jichu = '2018年全国等级考试成绩汇总及学籍信息\\学籍库模板'

data_xueji = pd.read_excel(os.path.join(data_path_jichu, '兰州理工大学在籍学籍库.xlsx'))
data_xuesheng = pd.read_excel(os.path.join(data_path_jichu, '在籍在校学生名单.xls'))

shenfen_dict = {"北京市": 11, "天津市": 12, "河北省": 13, "山西省": 14, "内蒙古自治区": 15,
                "辽宁省": 21, "吉林省": 22, "黑龙江省": 23,
                "上海市": 31, "江苏省": 32, "浙江省": 33, "安徽省": 34, "福建省": 35, "江西省": 36, "山东省": 37,
                "河南省": 41, "湖北省": 42, "湖南省": 43,
                "广东省": 44, "广西壮族自治区": 45, "海南省": 46,
                "重庆市": 50, "四川省": 51, "贵州省": 52, "云南省": 53, "西藏自治区": 54,
                "陕西省": 61, "甘肃省": 62, "青海省": 63, "宁夏回族自治区": 64, "新疆维吾尔自治区": 65,
                "台湾地区": 83, "香港特别行政区": 81, "澳门特别行政区": 82
                }
shenfen_dict_reverse = {v: k for k, v in shenfen_dict.items()}

kemu_dict = {'计算机基础及WPS Office应用': 14, '计算机基础及MS Office应用': 15, '计算机基础及Photoshop应用': 16,
             'C语言程序设计': 24, 'VB语言程序设计': 26, 'VFP数据库程序设计': 27, 'Java语言程序设计': 28,
             'Access数据库程序设计': 29,
             'C++语言程序设计': 61, 'MySQL数据程序设计': 63, 'Web程序设计': 64, 'MS Office高级应用': 65,
             '网络技术': 35, '数据库技术': 36, '软件测试技术': 37, '信息安全技术': 38, '嵌入式系统开发技术': 39,
             '网络工程师': 41, '数据库工程师': 42, '软件测试工程师': 43, '信息安全工程师': 44, '嵌入式系统开发工程师': 45}

kemu_dict_reverse = {v: k for k, v in kemu_dict.items()}

dengji_dict = {'计算机基础及WPS Office应用': '一', '计算机基础及MS Office应用': '一', '计算机基础及Photoshop应用': '一',
               '网络安全素质教育': '一', '公共基础知识': '二', 'C语言程序设计': '二', 'VB语言程序设计': '二',
               'Java语言程序设计': '二', 'Access数据库程序设计': '二', 'C++语言程序设计': '二', 'MySQL数据库程序设计': '二',
               'Web程序设计': '二', 'Python语言程序设计': '二', 'MS Office高级应用': "二",
               '网络技术': '三', '数据库技术': '三', '信息安全技术': '三', '嵌入式系统开发技术': '三',
               '操作系统原理': '四', '计算机组成与接口': '四', '计算机网络': '四', '数据库原理': '四'}


def contact_data():
    data_benbu = pd.read_excel(os.path.join(data_path_3, '2018年3月校本部成绩.xlsx'))
    data_xixiaoqu = pd.read_excel(os.path.join(data_path_3, '2018年3月西校区成绩.xlsx'))
    data_ncre_lut_3 = pd.concat([data_benbu, data_xixiaoqu], axis=0)
    data_ncre_lut_3.to_excel("data_ncre_lut_3.xlsx", index=None)
    print(data_ncre_lut_3.shape)
    data_benbu = pd.read_excel(os.path.join(data_path_9, '2018年9月校本部具体成绩.xlsx'))
    data_xixiaoqu = pd.read_excel(os.path.join(data_path_9, '2018年9月西校区具体成绩.xlsx'))
    data_ncre_lut_9 = pd.concat([data_benbu, data_xixiaoqu], axis=0)
    data_ncre_lut_9.to_excel("data_ncre_lut_9.xlsx", index=None)
    print(data_ncre_lut_9.shape)


def get_xueyuan(input):
    try:
        res = data_xueji[data_xueji['XM'].isin([input])]['FY'].values[0]
        return res
    except:
        res = data_xuesheng[data_xuesheng['姓名'].isin([input])]['学院'].values[0]
        return res


def get_xuehao(input):
    try:
        res = data_xueji[data_xueji['XM'].isin([input])]['XH'].values[0]
        return res
    except:
        res = data_xuesheng[data_xuesheng['姓名'].isin([input])]['学号'].values[0]
        return res


def get_shengfen(input):
    try:
        res = data_xueji[data_xueji['XM'].isin([input])]['SFZH'].values[0]
        return shenfen_dict_reverse[int(res[:2])]
    except:
        res1 = data_xuesheng[data_xuesheng['姓名'].isin([input])]['身份证号'].values[0]
        return shenfen_dict_reverse[int(res1[:2])]


def build_data():
    save = []
    # 姓名 性别 学号 学院 年级 学期 成绩 语种
    data_ncre_lut_3 = pd.read_excel("data_ncre_lut_3.xlsx")
    data_ncre_lut_9 = pd.read_excel("data_ncre_lut_9.xlsx")
    print(data_ncre_lut_3.shape)
    print(data_ncre_lut_9.shape)

    for x, y in data_ncre_lut_3.iterrows():
        try:
            rows = OrderedDict()
            rows['names'] = y['XM']
            rows['sex'] = y['XB']
            rows['province'] = get_shengfen(y['XM'])
            rows['xueyuan'] = get_xueyuan(y['XM'])
            rows['nianji'] = get_xuehao(y['XM'])[:2]
            rows['xueqi'] = '春季'
            rows['chengji'] = y['CJ']
            rows['yuzhong'] = kemu_dict_reverse[int(str(y['ZKZH'])[:2])]
            rows['dengji'] = dengji_dict[kemu_dict_reverse[int(str(y['ZKZH'])[:2])]]
            save.append(rows)
        except:
            continue

    print(len(save), '-------------1')

    for x, y in data_ncre_lut_9.iterrows():
        try:
            rows = OrderedDict()
            rows['names'] = y['xm']
            rows['sex'] = y['xb']
            rows['province'] = get_shengfen(y['xm'])
            rows['xueyuan'] = get_xueyuan(y['xm'])
            rows['nianji'] = get_xuehao(y['xm'])[:2]
            rows['xueqi'] = '秋季'
            rows['chengji'] = y['cj']
            rows['yuzhong'] = kemu_dict_reverse[int(str(y['zkzh'])[:2])]
            rows['dengji'] = dengji_dict[kemu_dict_reverse[int(str(y['zkzh'])[:2])]]
            save.append(rows)
        except Exception as e:
            continue

    data_all = pd.DataFrame(save)
    data_all.to_excel("全部需要数据.xlsx", index=None)
    print(data_all.shape)


if __name__ == "__main__":

    method = 'analysis'

    if method == 'contact_data':
        contact_data()

    if method == 'build_data':

        build_data()
        # (9819, 9)

    if method == 'analysis':
        """
        2.需要分析各个学院各个等级（如二级，三级，四级）的平均分，通过率；
        各个学院各个等级语种（如二级c语言/office...三级数据库/计算机网络技术...）的平均分以及通过率；
        各个年级各个等级的平均分和通过率；
        2018春季学期和秋季学期对比分析。
        3，将数据用图表，散点图可视化出来
        4.全校各省学生二级的通过率，用中国地图散点图表示出来
        """

        data = pd.read_excel("全部需要数据.xlsx")
        for x, y in data.groupby(by='yuzhong'):
            for x1, y1 in y.groupby(by='xueyuan'):
                for x2, y2 in y1.groupby(by='dengji'):
                    scores = y2['chengji'].mean()
                    lelve = [i for i in y2['chengji'] if i > 60]
                    rate = len(lelve)/y2.shape[0]
                    if lelve:
                        print(x, x1, x2, "{:0.3f}".format(scores), "{:0.3f}".format(rate))

        for x, y in data.groupby(by='xueqi'):
            for x1, y1 in y.groupby(by='nianji'):
                for x2, y2 in y1.groupby(by='dengji'):
                    scores = y2['chengji'].mean()
                    lelve = [i for i in y2['chengji'] if i > 60]

        for x, y in data.groupby(by='province'):
            for x2, y2 in y.groupby(by='dengji'):
                if '二' in x2:
                    lelve = [i for i in y2['chengji'] if i > 60]

