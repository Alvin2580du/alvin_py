import itchat
from itchat.content import TEXT, MAP, CARD, NOTE, SHARING
import time
from pyduyp.datasources.xiaohua import getone, update
import datetime

itchat.auto_login(hotReload=True)


def get_message_from_es(name, msg):
    search = getone()
    reply = search['name']
    updateinputs = {"name": search['name'], "status": 1, "createtime": datetime.datetime.now()}
    update(idstr=search['id'], inputs_dict=updateinputs)
    if name == '暖酒':
        res = "亲爱的，给你讲个笑话 \n\n ************************** {} \n\n\n\n".format(msg, reply)
        return res
    else:
        return "************************** \n\n\n {} \n\n\n**************************".format(reply)


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    print(msg['Content'])
    sendtime = msg.CreateTime
    user = msg['User']
    name = user['NickName']

    if name == '暖酒':
        timecost = time.time()-sendtime

        return get_message_from_es(name, msg.text)
    else:

        return get_message_from_es(name, msg.text)


itchat.run()
