import requests
import pandas as pd
import re
from sqlalchemy import Table, Column, VARCHAR, INT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, MetaData
import logging
from functools import wraps

asia_ch2num = {'平手': 0, '半球': 0.5, '一球': 1, '球半': 1.5, '两球': 2, '两球半': 2.5,
               '三球': 3, '三球半': 3.5, '四球': 4, '四球半': 4.5, '五球': 5, '五球半': 5.5, '六球': 6, '六球半': 6.5,
               '七球': 7, '七球半': 7.5, '八球': 8, '八球半': 8.5, '九球': 9, '九球半': 0.5, '十球': 10}


# 删除<>号
def dele(x):
    if re.search('<.*>', x):
        return re.sub('<.*>', '', x)
    else:
        return x


# 亚盘盘口汉字转数字
def ch_to_num(st):
    if st.__class__ == float:
        return None
    weight = -1 if '受让' in st else 1
    tem_st = st.split('/')
    sum = 0
    for i in range(len(tem_st)):
        tem_st[i] = tem_st[i].replace('受让', '')
        sum += asia_ch2num[tem_st[i]]

    return sum / weight / len(tem_st)


# 爬取欧盘数据，使用json数据爬取
def get_euro_odds(code):
    scheduleId = str(code)
    # 1674531
    # headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    url = 'http://1x2d.win007.com/' + scheduleId + '.js'
    res = requests.get(url)
    if res.status_code != 200:
        return [None] * 6

    europe = res.content.decode('utf-8')
    if re.search('William Hill.*?"', europe):
        temlist = re.search('William Hill.*?"', europe).group().split('|')
        temlist2 = []
        for x in (temlist[1:4] + temlist[8:11]):
            if x:
                temlist2.append(float(x))
            else:
                temlist2.append(None)
        return temlist2
    else:
        return [None] * 6


# 爬取亚盘数据，访问网页源代码
def get_asia_odds(code):
    url = 'http://vip.win007.com/AsianOdds_n.aspx?id=' + str(code)
    headers1 = {'Referer': 'http://vip.win007.com/AsianOdds_n.aspx?id=1662649',
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    res = requests.get(url, headers=headers1)
    asia_html = res.content.decode('gbk')

    if not re.search('<td height="25">Bet365.*?<a', asia_html, re.S):
        return [None] * 6
    tem_st = re.search('<td height="25">Bet365.*?<a', asia_html, re.S).group()
    if re.findall('>[\d\.]+?<', tem_st):
        asia_data = [float(x[1:-1]) for x in re.findall('>[\d\.]+?<', tem_st)]
        start_awin = asia_data[0]
        if not start_awin:
            print("start_awin", asia_data, start_awin)
        start_alost = asia_data[1]
        end_awin = asia_data[4]
        end_alost = asia_data[5]

        pankou_data = [x[1:-1] for x in re.findall(u'>[\u4e00-\u9fa5\/]+<', tem_st)]
        st_start_asia = pankou_data[0]
        start_asia = ch_to_num(st_start_asia)

        st_end_asia = pankou_data[-1]
        end_asia = ch_to_num(st_end_asia)

        asia_data = [start_awin, start_asia, start_alost, end_awin, end_asia, end_alost]
        return asia_data
    else:
        return [None] * 6


##############################################################################################
'''
抓取所有比赛
'''
headers = {'Cookie': 'win007BfCookie=null; bfWin007FirstMatchTime=2019,6,6,08,00,00',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36',
           'Host': 'live.win007.com',
           'Referer': 'http://live.win007.com/',
           'Accept-Encoding': 'gzip'}
url = 'http://live.win007.com/vbsxml/bfdata.js?r=0071562390372000'
res = requests.get(url, headers=headers)
data = res.content.decode('gbk').split('\n')

for i in range(len(data)):
    data[i] = data[i].split('^')
match_infor = []
for i in data:
    if 'A[' in i[0]:
        match_infor.append(i)

# 联盟名称，主队，客队,主队进球，客队进球，赛果
list_data = []
ind = 0
print('共有%d场比赛' % len(match_infor))
for i in match_infor:
    # i=match_infor[172]
    kind = dele(i[2])
    home_name = dele(i[5])
    quest_name = dele(i[8])
    content = home_name + ' VS ' + quest_name
    host = i[14]
    quest = i[15]

    if host == quest:
        mactch_result = '平'
    elif host > quest:
        mactch_result = '主胜'
    else:
        mactch_result = '主负'
    code = re.search('\d{7}', i[0]).group()
    # code='1711571'

    temlist_1 = [int(code), kind, content, home_name, quest_name, int(host), int(quest), mactch_result]
    euro_odds = get_euro_odds(code)
    temlist_1.extend(euro_odds)

    aisa_odds = get_asia_odds(code)
    temlist_1.extend(aisa_odds)

    list_data.append(temlist_1)
    ind += 1
    print('{}/{}'.format(ind, len(match_infor)))

final_data = pd.DataFrame(list_data)
final_data.columns = ['id', 'kind', 'content', 'home_name', 'guest_name',
                      'host', 'guest', 'mactch_result',
                      'start_win', 'start_draw', 'start_lost', 'end_win', 'end_draw', 'end_lost',
                      'start_awin', 'start_asia', 'start_alost', 'end_awin', 'end_asia', 'end_alost']

final_data['date'] = pd.datetime.today().strftime(format='%Y-%m-%d')

final_data.to_excel('result.xlsx', index=None)


args = {"db_type": 'mysql',
        "user": 'root',
        "password": 'mysql',
        "host": '127.0.0.1',
        "port": '3306',
        "db_name": 'alvin',
        }


def get_engine():
    connect_str = "{}+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(args['db_type'], args['user'], args['password'],
                                                                    args['host'], args['port'], args['db_name'])
    engine = create_engine(connect_str, encoding='utf-8')
    return engine


def get_db():
    eng = get_engine()
    Session = sessionmaker(bind=eng)
    db_session = Session()
    return db_session


db_session = get_db()

Base = declarative_base()

metadata = MetaData(get_engine())

tablename = 'statistics'

statistics_info = Table(tablename, metadata,
                        Column("id", INT, primary_key=True),
                        Column("kind", VARCHAR(200)),
                        Column("content", VARCHAR(200)),
                        Column("home_name", VARCHAR(200)),
                        Column("guest_name", VARCHAR(200)),
                        Column("mactch_result", VARCHAR(200)),
                        Column("predict_result", VARCHAR(200)),
                        Column("predict_result_short", VARCHAR(200)),
                        Column("predict_accurate", VARCHAR(200)),
                        Column("host", VARCHAR(200)),
                        Column("guest", VARCHAR(200)),
                        Column("start_win", VARCHAR(200)),
                        Column("start_draw", VARCHAR(200)),
                        Column("end_win", VARCHAR(200)),
                        Column("end_draw", VARCHAR(200)),
                        Column("end_lost", VARCHAR(200)),

                        Column("start_awin", VARCHAR(200)),

                        Column("start_asia", VARCHAR(200)),
                        Column("start_alost", VARCHAR(200)),
                        Column("end_asia", VARCHAR(200)),
                        Column("end_awin", VARCHAR(200)),
                        Column("end_alost", VARCHAR(200)),
                        Column("fenbu", VARCHAR(200)),
                        Column("rid", VARCHAR(200)),
                        Column("kid", VARCHAR(200)),
                        Column("evidence", VARCHAR(200)),
                        Column("isImage", VARCHAR(200)),
                        Column("date", VARCHAR(200)),
                        )


class Room(Base, dict):
    __table__ = statistics_info

    def __init__(self, dic):
        for key, val in dic.items():
            self.__dict__[key] = self[key] = Room(val) if isinstance(val, dict) else val


def db_commit_decorator(func):
    @wraps(func)
    def session_commit(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error('db operation error, detail {}'.format(e))
            logging.warning('transaction rollbacks')
            db_session.rollback()

    return session_commit


@db_commit_decorator
def get_bot_room_exist(room_id):
    sql = "select * from {} where content='{}'limit 1".format(tablename, room_id)
    logging.debug("get query sql:\n {}".format(sql))
    ret = None
    try:
        row = db_session.execute(sql).fetchall()
        if row is not None and len(row) > 0:
            ret = Room(dict(row[0]))
    except Exception as e:
        logging.warning("select error: {}".format(str(e)))
    db_session.close()
    return ret


@db_commit_decorator
def add_bot_room(bot_room):
    roomObj = get_bot_room_exist(bot_room['content'])
    if roomObj is None:
        try:
            bot_question_obj = Room(bot_room)
            db_session.add(bot_question_obj)
            db_session.commit()
            db_session.flush()
            db_session.refresh(bot_question_obj)
            bot_room = bot_question_obj
            logging.info("{}".format(bot_room))
        except Exception as e:
            logging.warning('add bot room error: {}'.format(str(e)))
            bot_room = None
        db_session.close()
    else:
        bot_room = roomObj
    return bot_room


for x, y in final_data.iterrows():
    print(y['end_draw'], type(y['end_draw']))

    botRoom = {'kind': y['kind'], 'content': y['content'],
               'home_name': y['home_name'], 'guest_name': y['guest_name'],
               'mactch_result': y['mactch_result'], 'predict_result': None,
               'predict_result_short': None, 'predict_accurate': None,
               'host': y['host'], 'guest': y['guest'],
               'start_win': y['start_win'], 'start_draw': y['start_draw'],
               'end_win': y['end_win'], 'end_draw': y['end_draw'],
               'end_lost': y['end_lost'], 'start_asia': y['start_asia'],
               'start_awin': y['start_awin'],
               'start_alost': y['start_alost'], 'end_asia': y['end_asia'],
               'end_awin': y['end_awin'], 'end_alost': y['end_alost'],
               'fenbu': None, 'rid': None,
               'kid': None, 'evidence': None, 'isImage': None,
               'date': y['date']}

    roomObj = add_bot_room(botRoom)



