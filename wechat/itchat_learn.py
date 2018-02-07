import re
import itchat
from itchat.content import TEXT, MAP, CARD, NOTE, SHARING, PICTURE, RECORDING, ATTACHMENT, VIDEO, FRIENDS
import time


@itchat.msg_register([TEXT, PICTURE, FRIENDS, CARD, MAP, SHARING, RECORDING, ATTACHMENT, VIDEO], isFriendChat=True,
                     isGroupChat=False, isMpChat=False)
def personal_msg(msg):
    msg_time_rec = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    msg_from = itchat.search_friends(userName=msg['FromUserName'])['NickName']

    msg_time = msg['CreateTime']

    msg_id = msg['MsgId']

    msg_content = None

    msg_share_url = None
    print(msg['Type'])
    print("== "*20)
    if msg['Type'] == 'Text' or msg['Type'] == "Friends":

        print(msg_time_rec + "  " + msg_from + ' send a ' + msg['Type'] + ' : ' + msg['Text'])

    elif msg['Type'] == 'Map':

        x, y, location = re.search("<location x=\"(.*?)\" y=\"(.*?)\".*label=\"(.*?)\".*", msg['OriContent']).group(1,
                                                                                                                    2,
                                                                                                                    3)

        print(msg_time_rec + "  " + msg_from + ' send a ' + msg['Type'] + ' : ' + msg['Text'])

        print('The detail location is : ' + u"纬度->" + x.__str__() + u" 经度->" + y.__str__())

    elif msg['Type'] == 'Card':

        print(msg_time_rec + "  " + msg_from + ' send a ' + msg['Type'] + ' : ' + msg['RecommendInfo'][
            'NickName'] + u'的名片')

    elif msg['Type'] == 'Sharing':

        print(msg_time_rec + "  " + msg_from + ' send a ' + msg['Type'] + ' : ' + msg['Text'])

        print('The Url is: ' + msg['Url'])

    elif msg['Type'] == "Attachment" or msg['Type'] == "Video" or msg['Type'] == 'Picture' or msg[
        'Type'] == 'Recording':

        print(msg_time_rec + "  " + msg_from + ' send a ' + msg['Type'])


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    # msg.user.send('%s: %s' % (msg.type, msg.text))
    reply_message = "我现在正忙，稍后回复~ "
    msg.user.send("{}".format(reply_message))

itchat.auto_login(hotReload=True)

# 运行程序

itchat.run()