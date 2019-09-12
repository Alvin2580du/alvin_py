import pandas as pd
import datetime
import json
from collections import OrderedDict

"""
题目一文件中，给了两列数据，第一列gid 是主键，第二列raw1。根据raw1中的数据，生成4 个新的字段，
包括注册总次数（每一行中不同的注册用分号隔开）、注册总次数_非银行、注册最早时间、注册平台数（每一行中不同的平台代码数）。
"""


def all_register(inputs):
    return len(inputs.split(";"))


# 平台类型:非银行,平台代码:GEO_0000004232,注册时间:2017-04-14;
def no_bank(inputs):
    num = 0
    inputs_sp = inputs.split(";")
    for item in inputs_sp:
        item_sp = item.split(",")[0]
        if item_sp == '平台类型:非银行':
            num += 1
    return num


def get_time(inputs):
    inputs_sp = inputs.split(";")
    begin = datetime.datetime.strptime('2100-01-01', '%Y-%m-%d')

    for item in inputs_sp:
        item_sp = item.split(",")[-1].split(":")[1]
        dateTime_p = datetime.datetime.strptime(item_sp, '%Y-%m-%d')
        if dateTime_p < begin:
            begin = dateTime_p
    return begin


def get_pingtai(inputs):
    inputs_sp = inputs.split(";")
    pintais = []

    for item in inputs_sp:
        item_sp = item.split(",")[1].split(":")[1]
        pintais.append(item_sp)

    return len(list(set(pintais)))


def question_one():
    data_one = pd.read_excel("题目一.xlsx")
    data_one['注册总次数'] = data_one['raw1'].apply(all_register)
    data_one['注册次数_非银行'] = data_one['raw1'].apply(no_bank)
    data_one['最早注册时间'] = data_one['raw1'].apply(get_time)
    data_one['注册平台数'] = data_one['raw1'].apply(get_pingtai)
    data_one.to_excel("题目一-结果.xlsx", index=None)
    print('success')


"""
把题目二文件中data1、data2、data3、data4 中的json 数据解析数来，分别放到4 个Excel 文件里面。每个文件中都保留主键gid。
"""
data_two = pd.read_excel("题目二.xlsx")

data1 = data_two['data1']


def complie_data(inputs):
    province = inputs['ISPNUM']['province']
    city = inputs['ISPNUM']['city']
    isp = inputs['ISPNUM']['isp']
    RSL = inputs['RSL']
    ECL = inputs['ECL']
    return province, city, isp, RSL, ECL


def comploe_RSL(inputs):
    try:
        code = inputs[0]['RS']['code']
        desc = inputs[0]['RS']['desc']
        IFT = inputs[0]['IFT']
        return code, desc, IFT
    except:
        return '', '', ''


def build_data1():
    save = []
    gid = 0
    for items in data1:
        gid += 1
        items2json = json.loads(items)
        code = items2json['code']
        data1_data = items2json['data']
        province, city, isp, RSL, ECL = complie_data(data1_data)
        RSL_code, desc, IFT = comploe_RSL(RSL)
        msg = items2json['msg']
        rows = OrderedDict()
        rows['gid'] = gid
        rows['code'] = code
        rows['province'] = province
        rows['city'] = city
        rows['isp'] = isp
        rows['RSL_code'] = RSL_code
        rows['desc'] = desc
        rows['IFT'] = IFT
        rows['msg'] = msg
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("题目二-data1.xlsx", index=None)
    print(df.shape)


def get_sign(inputs):
    try:
        sign = json.loads(inputs[0])['sign']
        return sign
    except:
        return ''


def get_header(inputs):
    try:
        sign = json.loads(inputs[0])['header']
        return sign
    except:
        return ''


def get_body(inputs):
    try:
        sign = json.loads(inputs[0])['body']
        return sign
    except:
        return ''


def get_result(inputs):
    try:
        if len(inputs) == 2:
            result = json.loads(inputs[1])['result']
            return result
        else:
            result = json.loads(inputs[2])['result']
            return result
    except:
        return ''


def get_CODEs(inputs):
    try:
        if len(inputs) == 3:
            CODE = json.loads(inputs[1])['CODE']
            PHONE = json.loads(inputs[1])['PHONE']
            PROVINCE = json.loads(inputs[1])['PROVINCE']
            CITY = json.loads(inputs[1])['CITY']
            RESULTS = json.loads(inputs[1])['RESULTS']
            return CODE, PHONE, PROVINCE, CITY, RESULTS
        else:
            return '', '', '', '', ''
    except:
        return '', '', '', '', ''


def complie_header(inputs):
    if inputs:
        resp_time = inputs['resp_time']
        ret_msg = inputs['ret_msg']
        version = inputs['version']
        ret_code = inputs['ret_code']
        req_time = inputs['req_time']
    else:
        resp_time = ''
        ret_msg = ''
        version = ''
        ret_code = ''
        req_time = ''
    return resp_time, ret_msg, version, ret_code, req_time


def complie_body(inputs):
    if inputs:
        MODEL7 = inputs['scores']['MODEL7']
        MODEL6 = inputs['scores']['MODEL6']
        MODEL5 = inputs['scores']['MODEL5']
        MODEL4 = inputs['scores']['MODEL4']
        MODEL3 = inputs['scores']['MODEL3']
        MODEL2 = inputs['scores']['MODEL2']
        MODEL1 = inputs['scores']['MODEL1']
        ud_order_no = inputs['ud_order_no']
    else:
        MODEL7 = ''
        MODEL6 = ''
        MODEL5 = ''
        MODEL4 = ''
        MODEL3 = ''
        MODEL2 = ''
        MODEL1 = ''
        ud_order_no = ''

    return MODEL7, MODEL6, MODEL5, MODEL4, MODEL3, MODEL2, MODEL1, ud_order_no


def complie_result(inputs):
    try:
        status = inputs['status']
        score = inputs['score']
        features = inputs['features']
        cid = inputs['cid']
    except:
        status = ''
        score = ''
        features = ''
        cid = ''

    return status, score, features, cid


def build_data2():
    data1 = data_two['data2']
    save = []
    gid = 0
    for items in data1:
        gid += 1
        rows = OrderedDict()
        rows['gid'] = gid
        lines = items.split(";")
        sign = get_sign(lines)
        rows['sign'] = sign

        header = get_header(lines)
        resp_time, ret_msg, version, ret_code, req_time = complie_header(header)
        rows['resp_time'] = resp_time
        rows['ret_msg'] = ret_msg
        rows['version'] = version
        rows['ret_code'] = ret_code
        rows['req_time'] = req_time

        body = get_body(lines)
        MODEL7, MODEL6, MODEL5, MODEL4, MODEL3, MODEL2, MODEL1, ud_order_no = complie_body(body)
        rows['MODEL7'] = MODEL7
        rows['MODEL6'] = MODEL6
        rows['MODEL5'] = MODEL5
        rows['MODEL4'] = MODEL4
        rows['MODEL3'] = MODEL3
        rows['MODEL2'] = MODEL2
        rows['MODEL1'] = MODEL1
        rows['ud_order_no'] = ud_order_no

        result = get_result(lines)
        status, score, features, cid = complie_result(result)

        rows['status'] = status
        rows['score'] = score
        rows['features'] = features
        rows['cid'] = cid
        if result == '未命中':
            rows['result'] = "未命中"
        else:
            rows['result'] = ""

        CODE, PHONE, PROVINCE, CITY, RESULTS = get_CODEs(lines)

        rows['CODE'] = CODE
        rows['PHONE'] = PHONE
        rows['PROVINCE'] = PROVINCE
        rows['CITY'] = CITY
        rows['MODEL2'] = MODEL2
        rows['RESULTS'] = RESULTS
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("题目二-data2.xlsx", index=None)
    print(df.shape)


def complie_data3_data(inputs):
    try:
        inputs = json.loads(inputs.split(";")[0])['data']
        cg = inputs['cg']
        pr = inputs['pr']
        ap = inputs['ap']
        in_ = inputs['in']
        co = inputs['co']
        li = inputs['li']
        return cg, pr, ap, in_, co, li
    except:
        return '', '', '', '', '', ''


def complie_data3_result(item):
    try:
        inputs = item.split(";")
        result = json.loads(inputs[1])['result']
        if result == '未命中':
            return '未命中', '', '', '', ''
        else:
            status = result['status']
            score = result['score']
            features = result['features']
            cid = result['cid']
            return result, status, score, features, cid
    except:
        return '', '', '', '', ''


def complie_error(inputs):
    try:
        lines = inputs.split(";")
        errcode = json.loads(lines[0])['errcode']
        errdetail = json.loads(lines[0])['errdetail']
        errmsg = json.loads(lines[0])['errmsg']
        ext = json.loads(lines[0])['ext']
        return errcode, errdetail, errmsg, ext
    except:
        return '', '', '', ''


def build_data3():
    data3 = data_two['data3']

    save = []
    gid = 0
    for items in data3:
        gid += 1
        rows = OrderedDict()
        rows['gid'] = gid
        cg, pr, ap, in_, co, li = complie_data3_data(items)
        rows['cg'] = cg
        rows['pr'] = pr
        rows['ap'] = ap
        rows['in'] = in_
        rows['co'] = co
        rows['li'] = li
        errcode, errdetail, errmsg, ext = complie_error(items)

        rows['errcode'] = errcode
        rows['errdetail'] = errdetail
        rows['errmsg'] = errmsg
        rows['ext'] = ext

        result, status, score, features, cid = complie_data3_result(items)
        if result == '未命中':
            rows['result'] = '未命中'
        else:
            rows['result'] = ''

        rows['status'] = status
        rows['score'] = score
        rows['features'] = features
        rows['cid'] = cid
        save.append(rows)
    df = pd.DataFrame(save)
    df.to_excel("题目二-data3.xlsx", index=None)
    print(df.shape)


def build_data4():
    data4 = data_two['data4']
    save = []
    gid = 0
    for items in data4:
        items2json = json.loads(items)
        gid += 1
        rows = OrderedDict()
        rows['gid'] = gid
        rows['meid'] = items2json['meid']
        rows['city'] = items2json['city']
        rows['modelId'] = items2json['modelId']
        rows['latitude'] = items2json['latitude']
        rows['isp'] = items2json['isp']
        rows['regDate'] = items2json['regDate']
        rows['imsi'] = items2json['imsi']
        rows['idNumber'] = items2json['idNumber']
        rows['cardNo'] = items2json['cardNo']
        rows['entryId'] = items2json['entryId']
        rows['INDX304000'] = items2json['INDX304000']
        rows['sequenceNo'] = items2json['sequenceNo']
        rows['province'] = items2json['province']
        rows['workAdress'] = items2json['workAdress']
        rows['cid2'] = items2json['cid2']
        rows['company'] = items2json['company']
        rows['vin'] = items2json['vin']
        rows['applyTime'] = items2json['applyTime']
        rows['deadline'] = items2json['deadline']
        rows['longitude'] = items2json['longitude']
        rows['addressCode'] = items2json['addressCode']
        rows['original_cid'] = items2json['original_cid']
        rows['address'] = items2json['address']
        rows['dataType'] = items2json['dataType']
        rows['ip'] = items2json['ip']
        rows['original_cid2'] = items2json['original_cid2']
        rows['homeAdress'] = items2json['homeAdress']
        rows['usualaddress'] = items2json['usualaddress']
        rows['realName'] = items2json['realName']
        rows['platenumber'] = items2json['platenumber']
        rows['platetype'] = items2json['platetype']
        rows['month'] = items2json['month']
        rows['cid'] = items2json['cid']
        save.append(rows)

    df = pd.DataFrame(save)
    df.to_excel("题目二-data4.xlsx", index=None)


if __name__ == '__main__':

    method = 'build_data4'

    if method == 'question_one':
        question_one()

    if method == 'build_data1':
        build_data1()

    if method == 'build_data2':
        build_data2()

    if method == 'build_data3':
        build_data3()

    if method == 'build_data4':
        build_data4()

