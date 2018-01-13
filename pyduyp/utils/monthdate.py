# coding=utf-8

import datetime
import time
import re
from pyduyp.logger.log import log
from pyduyp.utils.utils import remove_empty
from pyduyp.extend.timeservice import postjson


def beforetonow(begin_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(time.time())), "%Y-%m-%d")
    while begin_date <= end_date:
        # date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(begin_date)
        begin_date += datetime.timedelta(days=1)
    return date_list


def beforeend(begin_date, end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    while begin_date <= end_date:
        date_list.append(begin_date)
        begin_date += datetime.timedelta(days=1)
    return date_list


def preday(s):
    theday = datetime.date(*map(int, s.split('-')))
    prevday = theday - datetime.timedelta(days=1)
    day = prevday.strftime('%Y-%m-%d')
    return day


def replaceDateStr(tStr, resultList, redate):
    return tStr.replace(redate, '|' + '|'.join(resultList) + '|')


def betweenDateTrans(tStr):
    '''
        针对正则多次匹配的情况做出修改
        将公用的部分尽量封装起来
    '''

    try:
        re0 = '\d+\-\d+\-\d+[至到\～\~]\d+\-\d+\-\d+'
        if re.findall(re0, tStr, re.S):
            reMatchList = re.findall(re0, tStr, re.S)

            for redate in reMatchList:
                tempDate = redate.replace('到', '至').replace('～', '至').replace('~', '至').strip()
                tempDateList = tempDate.split('至')
                if len(tempDateList) == 2:
                    resultList = []
                    startDate = tempDateList[0]
                    endDate = tempDateList[1]
                    startTime = datetime.datetime.strptime(startDate, '%Y-%m-%d')
                    endTime = datetime.datetime.strptime(endDate, '%Y-%m-%d')
                    while startTime < endTime:
                        formatTime = startTime.strftime('%m-%d')
                        formatTimeList = formatTime.split('-')
                        if len(formatTimeList) == 2:
                            dateItem = formatTimeList[0] + '月' + formatTimeList[1] + '日'
                            resultList.append(dateItem)
                            startTime = startTime + datetime.timedelta(days=1)
                    tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re01 = '\d+[\/]\d+[\-到至\～\~]\d+[\/]\d+[号日]'
        if re.findall(re01, tStr, re.S):
            reMatchList = re.findall(re01, tStr, re.S)
            for redate in reMatchList:
                tempDate = redate.replace('到', '至').replace('～', '至').replace('~', '至').replace('-', '至'). \
                    replace('号', '').replace('日', ''). \
                    replace('/', '.')
                redateSplitList = tempDate.split('至')
                if len(redateSplitList) == 2:
                    resultList = []
                    startTime = datetime.datetime.strptime(redateSplitList[0], '%m.%d')
                    endTime = datetime.datetime.strptime(redateSplitList[1], '%m.%d')
                    while startTime < endTime:
                        formatTime = startTime.strftime('%m-%d')
                        formatTimeList = formatTime.split('-')
                        if len(formatTimeList) == 2:
                            dateItem = formatTimeList[0] + '月' + formatTimeList[1] + '日'
                            resultList.append(dateItem)
                            startTime = startTime + datetime.timedelta(days=1)
                    tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        # re1 = '\d+月\d+日[\-|到|至|\～|\~]\d+月\d+日'
        re1 = '\d+月\d+日[\-到至\～\~]\d+月\d+日'
        re2 = '\d+[\.\,\，月\/]\d+[\-到\～至\~]\d+[\.\,\，月\/]\d+'
        if re.findall(re1 + '|' + re2, tStr, re.S):
            reMatchList = re.findall(re1 + '|' + re2, tStr, re.S)

            for redate in reMatchList:
                if re.search(re1, redate, re.S):
                    resultList = []
                    tempDate = redate.replace('到', '-').replace('至', '-').replace('～', '-').replace('~', '-').strip()
                    redateSplitList = tempDate.split('-')
                    if len(redateSplitList) == 2:
                        startDate = redateSplitList[0]
                        endDate = redateSplitList[1]
                        startTime = datetime.datetime.strptime(startDate, '%m月%d日')
                        endTime = datetime.datetime.strptime(endDate, '%m月%d日')
                        while startTime < endTime:
                            formatTime = startTime.strftime('%m-%d')
                            formatTimeList = formatTime.split('-')
                            if len(formatTimeList) == 2:
                                dateItem = formatTimeList[0] + '月' + formatTimeList[1] + '日'
                                resultList.append(dateItem)
                                startTime = startTime + datetime.timedelta(days=1)
                    tStr = replaceDateStr(tStr, resultList, redate)

                if re.search(re2, redate, re.S):
                    resultList = []
                    tempDate = redate.replace('到', '-').replace('～', '-').replace('至', '-').replace('~', '-') \
                        .replace(',', '.').replace('，', '.').replace('月', '.').replace('/', '.')
                    redateSplitList = tempDate.split('-')
                    if len(redateSplitList) == 2:
                        startTime = datetime.datetime.strptime(redateSplitList[0], '%m.%d')
                        endTime = datetime.datetime.strptime(redateSplitList[1], '%m.%d')
                        while startTime < endTime:
                            formatTime = startTime.strftime('%m-%d')
                            formatTimeList = formatTime.split('-')
                            if len(formatTimeList) == 2:
                                dateItem = formatTimeList[0] + '月' + formatTimeList[1] + '日'
                                resultList.append(dateItem)
                                startTime = startTime + datetime.timedelta(days=1)
                    tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re3 = '\d+月\d+[日号號][\-至到\～\~]\d+[日号號]'
        if re.findall(re3, tStr, re.S):
            reMatchList = re.findall(re3, tStr, re.S)

            for redate in reMatchList:
                resultList = []
                tempDate = redate.replace('到', '-').replace('～', '-').replace('至', '-').replace('~', '-') \
                    .replace('号', '').replace('日', '').replace('號', '')
                mothAndDayList = tempDate.split('月')
                dayList = mothAndDayList[1].split('-')
                startDate = mothAndDayList[0] + '月' + dayList[0] + '日'
                endDate = mothAndDayList[0] + '月' + dayList[1] + '日'
                startTime = datetime.datetime.strptime(startDate, '%m月%d日')
                endTime = datetime.datetime.strptime(endDate, '%m月%d日')
                while startTime < endTime:
                    formatTime = startTime.strftime('%m-%d')
                    formatTimeList = formatTime.split('-')
                    if len(formatTimeList) == 2:
                        dateItem = formatTimeList[0] + '月' + formatTimeList[1] + '日'
                        resultList.append(dateItem)
                        startTime = startTime + datetime.timedelta(days=1)
                tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re30 = '\d+月\d+[\-至到\～\~]\d+'
        if re.findall(re30, tStr, re.S):
            reMatchList = re.findall(re30, tStr, re.S)

            for redate in reMatchList:
                resultList = []
                tempDate = redate.replace('到', '-').replace('～', '-').replace('至', '-').replace('~', '-') \
                    .replace('号', '').replace('日', '').replace('號', '')
                mothAndDayList = tempDate.split('月')
                dayList = mothAndDayList[1].split('-')
                startDate = mothAndDayList[0] + '月' + dayList[0] + '日'
                endDate = mothAndDayList[0] + '月' + dayList[1] + '日'
                startTime = datetime.datetime.strptime(startDate, '%m月%d日')
                endTime = datetime.datetime.strptime(endDate, '%m月%d日')
                while startTime < endTime:
                    formatTime = startTime.strftime('%m-%d')
                    formatTimeList = formatTime.split('-')
                    if len(formatTimeList) == 2:
                        dateItem = formatTimeList[0] + '月' + formatTimeList[1] + '日'
                        resultList.append(dateItem)
                        startTime = startTime + datetime.timedelta(days=1)
                tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re4 = '\d+[到\-\～至\~]\d+[日号]'
        re5 = '\d+[号日][到至\-\～\~]\d+[号日]'
        re51 = '\d+[\-\~\～到至]\d+'
        if re.findall(re4 + '|' + re5 + '|' + re51, tStr, re.S):
            reMatchList = re.findall(re4 + '|' + re5 + '|' + re51, tStr, re.S)

            for redate in reMatchList:
                if re.search(re4, redate, re.S):
                    resultList = []
                    tempDate = redate.replace('到', '-').replace('～', '-').replace('至', '-').replace('~', '-') \
                        .replace('日', '').replace('号', '')
                    tempDateList = tempDate.split('-')
                    if len(tempDateList) == 2:
                        startNum = tempDateList[0]
                        endNum = tempDateList[1]
                        if int(endNum) > int(startNum):
                            for i in range(int(startNum), int(endNum) + 1):
                                # for i in range(int(startNum), int(endNum)):
                                if i > 31:
                                    continue
                                resultList.append(str(i) + '号')
                    tStr = replaceDateStr(tStr, resultList, redate)

                if re.search(re5, redate, re.S):
                    resultList = []
                    tempDate = redate.replace('到', '-').replace('～', '-').replace('至', '-').replace('~', '-') \
                        .replace('日', '').replace('号', '')
                    tempDateList = tempDate.split('-')
                    if len(tempDateList) == 2:
                        startNum = tempDateList[0]
                        endNum = tempDateList[1]
                        if int(endNum) > int(startNum):
                            # for i in range(int(numList[0]),int(numList[1])+1):
                            for i in range(int(startNum), int(endNum)):
                                resultList.append(str(i) + '号')
                    tStr = replaceDateStr(tStr, resultList, redate)

                if re.search(re51, redate, re.S):
                    resultList = []
                    tempDate = redate.replace('到', '-').replace('～', '-').replace('至', '-').replace('~', '-') \
                        .replace('日', '').replace('号', '')
                    tempDateList = tempDate.split('-')
                    if len(tempDateList) == 2:
                        startNum = tempDateList[0]
                        endNum = tempDateList[1]
                        if int(endNum) > int(startNum):
                            # for i in range(int(numList[0]),int(numList[1])+1):
                            for i in range(int(startNum), int(endNum)):
                                if i > 31:
                                    continue
                                resultList.append(str(i) + '号')
                    tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re6 = '\d+\.\d+\.\d+[号日]'
        # re7 = '\d{3,6}号'
        re8 = '\d+\.\d+\.\d+三.'
        # if re.findall(re6 + '|' + re7 + '|' + re8, tStr, re.S):
        if re.findall(re6 + '|' + re8, tStr, re.S):
            # reMatchList = re.findall(re6 + '|' + re7 + '|' + re8, tStr, re.S)
            reMatchList = re.findall(re6 + '|' + re8, tStr, re.S)

            for redate in reMatchList:
                # if re.search(re7, redate, re.S):
                #     resultList = []
                #     tempDate = redate.replace('号', '').strip()
                #     numList = []
                #     if len(tempDate) == 3:
                #         for i in range(len(tempDate)):
                #             numList.append(tempDate[i])
                #     if len(tempDate) == 6:
                #         numList.append(tempDate[:2])
                #         numList.append(tempDate[2:4])
                #         numList.append(tempDate[4:6])
                #     if len(tempDate) == 4:
                #         numList = [8, 9, 10]
                #     if len(tempDate) == 5:
                #         numList = [9, 10, 11]
                #     if numList:
                #         for num in numList:
                #             resultList.append(str(num) + '号')
                #     tStr = replaceDateStr(tStr, resultList, redate)

                if re.search(re6, redate, re.S):
                    resultList = []
                    numList = redate.replace('号', '').replace('日', '').strip().split('.')
                    if len(numList) == 3:
                        for num in numList:
                            if num > 31:
                                continue
                            resultList.append(str(num) + '号')
                    tStr = replaceDateStr(tStr, resultList, redate)

                if re.search(re8, redate, re.S):
                    resultList = []
                    numList = redate[:-2].strip().split('.')
                    if len(numList) == 3:
                        for num in numList:
                            if num > 31:
                                continue
                            resultList.append(str(num) + '号')
                    tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re81 = '\d+\.\d+[号|日]'
        if re.findall(re81, tStr, re.S):
            reMatchList = re.findall(re81, tStr, re.S)
            for redate in reMatchList:
                resultList = []
                tempDate = redate.replace('号', '').replace('日', '')
                num1 = int(tempDate.split('.')[0])
                num2 = int(tempDate.split('.')[1])
                if num1 + 1 == num2:  # 23.24号
                    resultList.append(str(num1) + '号')
                    resultList.append(str(num2) + '号')
                else:  # 7.18号
                    resultList.append(str(num1) + '月' + str(num2) + '日')
                tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re82 = '\d+[\,\，\,\、]\d+[号日]'
        if re.findall(re82, tStr, re.S):
            reMatchList = re.findall(re82, tStr, re.S)
            for redate in reMatchList:
                tempDate = redate.replace('号', '').replace('日', '') \
                    .replace('、', '.').replace(',', '.').replace('，', '.').replace(',', '.')
                tempDateList = tempDate.split('.')
                resultList = [dateItem + '号' for dateItem in tempDateList]
                tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        re9 = '\d+[和]\d+两.'
        if re.findall(re9, tStr, re.S):
            reMatchList = re.findall(re9, tStr, re.S)
            for redate in reMatchList:
                resultList = []
                numList = redate[:-2].replace('和', ',') \
                    .split(',')
                if len(numList) == 2:
                    for num in numList:
                        if num > 31:
                            continue
                        resultList.append(str(num) + '号')
                tStr = replaceDateStr(tStr, resultList, redate)
            return tStr

        # re10 = '\d+[\.\,\，\,]\d+[号日]'
        # if re.findall(re10, tStr, re.S):
        #     reMatchList = re.findall(re10, tStr, re.S)
        #
        #     for redate in reMatchList:
        #         resultList = []
        #         tempDate = redate.replace(',', ',').replace('，', ',').replace(',', ',').replace('.', ',') \
        #             .replace('号', '').replace('日', '')
        #         tempDateList = tempDate.split(',')
        #         if len(tempDateList) == 2:
        #             resultList.append(tempDateList[0] + '号')
        #             resultList.append(tempDateList[1] + '号')
        #         tStr = replaceDateStr(tStr, resultList, redate)

        # re11 = '\d+\.\d+'
        # if re.findall(re11, tStr, re.S):
        #     reMatchList = re.findall(re11, tStr, re.S)
        #     for redate in reMatchList:
        #         resultList = []
        #         tempDateList = redate.strip().split('.')
        #         if len(tempDateList) == 2:
        #             if int(tempDateList[0]) > 12:
        #                 resultList.append(tempDateList[0] + '日')
        #                 resultList.append(tempDateList[1] + '日')
        #             else:
        #                 resultList.append(tempDateList[0] + '月' + tempDateList[1] + '日')
        #         tStr = replaceDateStr(tStr, resultList, redate)


        re12 = '\d+[号日]'  # 1号 12号 30号 67号 234号 91011号 111213号 97号 32号 100号
        if re.findall(re12, tStr, re.S):
            reMatchList = re.findall(re12, tStr, re.S)
            for redate in reMatchList:
                resultList = []
                tempDate = redate.replace('号', '').replace('日', '')
                if len(tempDate) == 1:
                    resultList.append(str(tempDate) + '号')
                elif len(tempDate) == 2:
                    if int(tempDate) > 31 and int(tempDate[0]) + 1 == int(tempDate[1]):
                        resultList.append(tempDate[0] + '号')
                        resultList.append(tempDate[1] + '号')
                    elif int(tempDate) < 32:
                        resultList.append(tempDate + '号')
                elif len(tempDate) == 3:
                    # if int(tempDate[0]) + 1 == int(tempDate[1]) and \
                    #                         int(tempDate[1]) + 1 == int(tempDate[2]):
                    if int(tempDate[0]) < int(tempDate[1]) and \
                                    int(tempDate[1]) < int(tempDate[2]):
                        resultList.append(tempDate[0] + '号')
                        resultList.append(tempDate[1] + '号')
                        resultList.append(tempDate[2] + '号')
                # elif len(tempDate) == 4 and tempDate=='8910':
                elif len(tempDate) == 4:
                    if int(tempDate[0]) < int(tempDate[1]) and \
                                    int(tempDate[1]) < int(tempDate[2:]):
                        resultList.append(tempDate[0] + '号')
                        resultList.append(tempDate[1] + '号')
                        resultList.append(tempDate[2:] + '号')
                elif len(tempDate) == 5:
                    if int(tempDate[0]) < int(tempDate[1:3]) and \
                                    int(tempDate[1:3]) < int(tempDate[3:]):
                        resultList.append(tempDate[0] + '号')
                        resultList.append(tempDate[1:3] + '号')
                        resultList.append(tempDate[3:] + '号')
                elif len(tempDate) == 6:
                    if int(tempDate[0:2]) < int(tempDate[2:4]) and \
                                    int(tempDate[2:4]) < int(tempDate[4:]):
                        resultList.append(tempDate[0:2] + '号')
                        resultList.append(tempDate[2:4] + '号')
                        resultList.append(tempDate[4:] + '号')

                tStr = replaceDateStr(tStr, resultList, redate)

            return tStr
        return tStr
    except Exception as err:
        log.debug(err)
    return tStr


# 获取用户提到的时间
# 如果有房态的话，采取处理时间
def get_time_field_by_re(message):
    times = []
    timesre = [
        '\d+\-\d+\-\d+[至到\～\~]\d+\-\d+\-\d+',
        '\d+[\/]\d+[\-到至\～\~]\d+[\/]\d+[号日]',
        '\d+月\d+日[\-到至\～\~]\d+月\d+日',
        '\d+[\.\,\，月\/]\d+[\-到\～至\~]\d+[\.\,\，月\/]\d+',
        '\d+月\d+[日号號][\-至到\～\~]\d+[日号號]',
        '\d+月\d+[\-至到\～\~]\d+',
        '\d+[到\-\～至\~]\d+[日号]',
        '\d+[号日][到至\-\～\~]\d+[号日]',
        '\d+[\-\~\～到至]\d+',

        '\d{1,2}\.\d{1,2}\.\d{1,2}[号日]',
        '\d+\.\d+\.\d+三.',

        '\d{1,2}\.\d{1,2}[号日]',  # 这块误杀了 1.8米床 撒的，都出错了 这里有歧义：（1）23.24号（2）7.14号（3）1.8米不属于日期类别
        # '\d+\、\d+号',  # 新添加的1、2号 具体场景应该指天数
        # '\d[1-31]号$',  # 这个是 100号 出错，地址里面也有 100号，不是日期
        # '\d[一二三四五]号',

        '\d{1,2}[\,\，\,\、]\d{1,2}[号日]',  # \,\，\,\、表示两天  \.可能有歧义
        '\d+[和]\d+两.',

        '(今天|明天|后天|今晚)',
        '\d{1,2}[号日]'  # 1号 12号 30号
    ]
    try:
        # for timere in timesre:
        #     matches = re.findall(timere, message, re.S)
        #     if matches and len(matches) > 0:
        #         times.extend(matches)
        #         break

        for timere in timesre:
            reList = re.findall(timere, message, re.S)
            if reList:
                times.extend(reList)
                for reItem in reList:
                    message = message.replace(reItem, '')
    except Exception as err:
        log.warn(err)
    return times


# 获取用户提到的时间
# 如果有房态的话，采取处理时间
def get_time_field_by_re__backup(message):
    times = []
    timesre = [
        '\d+\-\d+\-\d+[至到\～\~]\d+\-\d+\-\d+',
        '\d+[\/]\d+[\-到至\～\~]\d+[\/]\d+[号日]',
        '\d+月\d+日[\-到至\～\~]\d+月\d+日',
        '\d+[\.\,\，月\/]\d+[\-到\～至\~]\d+[\.\,\，月\/]\d+',
        '\d+月\d+[日号號][\-至到\～\~]\d+[日号號]',
        '\d+月\d+[\-至到\～\~]\d+',
        '\d+[到\-\～至\~]\d+[日号]',
        '\d+[号日][到至\-\～\~]\d+[号日]',
        '\d+\、\d+号',  # 新添加的1、2号
        '\d+\.\d+\.\d+[号日]',
        '\d+[\-]d+',
        '\d+\-\d+',
        '\d+\.\d+号',  # 这块误杀了 1.8米床 撒的，都出错了
        '\d[1-31]号$',  # 这个是 100号 出错，地址里面也有 100号，不是日期
        # '\d[一二三四五]号',
        '\d+\.\d+\.\d+三.',
        '\d+[和]\d+两.',
        '\d+[\.\,\，\,]\d[号日]',
        '(今天|明天|后天)',
        '\d+[号日]',
        # '\d+\-d+[号日]',
        # '\d+\.\d+[号日]',

    ]
    try:
        for timere in timesre:
            matches = re.findall(timere, message, re.S)
            if matches and len(matches) > 0:
                times.extend(matches)
                break
            if len(times) > 90:   # 限定最长是三个月的
                break
    except Exception as err:
        log.warn(err)
    return times


def get_message_time_timelist(message):
    alldate = []
    times = get_time_field_by_re(message)
    # log.debug(times)
    for date in times:
        # 把时间处理对了，就可以不用调用时间服务了
        datelist = remove_empty(betweenDateTrans(date).split('|'))
        alldate.extend(datelist)
    return times, alldate


def get_standard_time(times, isDebug=False):
    alldate = []
    for date in times:
        log.debug(date)
        log.debug(betweenDateTrans(date).split('|'))
        datelist = remove_empty(betweenDateTrans(date).split('|'))
        # 去时间服务获取当前时间
        log.debug(datelist)
        datelist = postjson(datelist)
        log.debug(datelist)
        for d in datelist:
            if d not in alldate:
                # TODO: 如果时间小于今天就不添加
                # if time_validate_today(d):
                alldate.append(d)
    return alldate


def uniq_extend(alldate, datelist):
    for d in datelist:
        if d not in alldate:
            alldate.append(d)
    return alldate


def time_validate_today(check_date):
    check_dates = datetime.datetime.strptime(check_date, "%Y-%m-%d")
    today = datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d'), "%Y-%m-%d")
    delta = check_dates - today
    if delta.days >= 0:
        return True
    return False


# 计算2个时间字符串的时间差
def compute_date_interval(str1, str2, format="%Y-%m-%d %H:%M:%S"):
    try:
        time1 = datetime.datetime.strptime(str1, format)
        time2 = datetime.datetime.strptime(str2, format)
        if time2 < time1:
            time2, time1 = time1, time2
        ret = (time2 - time1).seconds
        return ret
    except Exception as e:
        log.warn(e)
        log.warn("parmas input error: {} {}".format(str1, str2))
        return 0


def cal_list_time_avg(timelist):
    size = len(timelist)
    if size < 2:
        return 0
    total = 0
    log.debug("size: {}".format(size))
    for i in range(size):
        if i + 1 >= size:
            break
        val = compute_date_interval(timelist[i], timelist[i + 1])
        log.debug("{}-{} {} {} avg:{}".format(i, i + 1, timelist[i], timelist[i + 1], val))

        total += val
    return round(total / size)


def cal_list_time_avg_pair(timelist):
    size = len(timelist)
    if size < 2:
        return 0
    total = 0
    i = 0
    max = 0
    min = 999
    while i < size - 1:
        # log.debug("range:{}-{} {} {}".format(i, i + 1, timelist[i], timelist[i+1]))
        t = compute_date_interval(timelist[i], timelist[i + 1])
        total += t
        if t > max:
            max = t
        if t < min:
            min = t
        i += 2
        if i >= size:
            break
    return round(total / (size / 2)), max, min
