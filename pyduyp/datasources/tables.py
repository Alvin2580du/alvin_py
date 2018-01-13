# coding=utf-8
"""
all tabls
author: alvin
"""
from sqlalchemy import Table, Column, INTEGER, String, BIGINT, TIMESTAMP, DATETIME, SMALLINT
from pyduyp.datasources.basic_db import metadata

import enum


class EstateEnum(enum.Enum):
    'valid',
    'deleted'


#
user_info = Table("user", metadata,
                  Column("id", BIGINT, primary_key=True),
                  Column("username", String(256)),
                  Column("password", String(256)),
                  Column("nickname", String(256)),
                  Column("realname", String(256)),
                  Column("provinceid", INTEGER, default=0),
                  Column("cityid", INTEGER, default=0),
                  Column("email", String(256)),
                  Column("mobile", String(256)),
                  Column("areacode", String(10), default='86'),
                  Column("sex", String(20)),
                  Column("headimageurl", String(256)),
                  Column("serverid", INTEGER, default=0),
                  #Column("estate", Enum(EstateEnum)),
                  Column("ver", INTEGER),
                  Column("remark", String(256)),
                  Column("createtime", DATETIME),
                  Column("updatetime", DATETIME),
                  Column("lastvisittime", DATETIME),
                  Column("property", INTEGER),
                  Column("fastpaystate", SMALLINT, default=0),
                  Column("fastpayforbidstarttime", DATETIME),
                  Column("fastpayforbidendtime", DATETIME),
                  Column("fastpayrefusereason", String(256)),
                  Column("fastpaypermittime", DATETIME),
                  Column("othermobile", String(256)),
                  Column("otherareacode", String(10)),
                  Column("noticestate", INTEGER),
                  Column("age", INTEGER),
                  Column("welcomes", String(256)),
                  Column("profession", String(256)),
                  Column("introduce", String(1024)),
                  Column("type", INTEGER),
                  Column("certno", String(20)),
                  Column("certname", String(20)),
                  Column("constellation", INTEGER),
                  Column("bloodtype", String(4)),
                  Column("housetown_province", INTEGER),
                  Column("housetown_city", INTEGER),
                  Column("identifyauth", INTEGER),
                  Column("papertype", INTEGER),
                  Column("countryid", INTEGER),
                  Column("paperno", String(50)),
                  Column("nationid", INTEGER),
                  Column("uploadstate", String(4), default='1'),
                  Column("lastremark", String(512)),
                  Column("companyid", BIGINT),
                  Column("wirter", INTEGER, default=0),
                  )


im_msgs_30_info = Table("im_msgs_30", metadata,
                     Column("id", BIGINT, primary_key=True),
                     Column("from", BIGINT),
                     Column("to", BIGINT),
                     Column("talkId", BIGINT),
                     Column("roomId", BIGINT),
                     Column("userType", String(10)),
                     Column("msgType", String(10)),
                     Column("msg", String(3000)),
                     Column("platform", String(10)),
                     Column("version", String(10)),
                     Column("state", INTEGER),
                     Column("type", INTEGER, default=0),
                     Column("opType", INTEGER, default=0),
                     Column("createTime", TIMESTAMP),
                     Column("uniqid", String(200)),
                     Column("channel", String(50)),
                     )


im_msgs_info = Table("im_msgs", metadata,
                     Column("id", BIGINT, primary_key=True),
                     Column("from", BIGINT),
                     Column("to", BIGINT),
                     Column("talkId", BIGINT),
                     Column("roomId", BIGINT),
                     Column("userType", String(10)),
                     Column("msgType", String(10)),
                     Column("msg", String(3000)),
                     Column("platform", String(10)),
                     Column("version", String(10)),
                     Column("state", INTEGER),
                     Column("type", INTEGER, default=0),
                     Column("opType", INTEGER, default=0),
                     Column("createTime", TIMESTAMP),
                     Column("uniqid", String(200)),
                     Column("channel", String(50)),
                     )


user_frozen_account_info = Table("user_frozen_account", metadata,
                     Column("id", BIGINT, primary_key=True),
                     Column("userid", BIGINT),
                     Column("username", String(256)),
                     Column("mobile", String(256)),
                     Column("paytype", String(20)),
                     Column("account", String(256)),
                     Column("state", INTEGER),
                     Column("operid", BIGINT),
                     Column("opername", String(256)),
                     Column("frozenreason", String(256)),
                     Column("createtime", TIMESTAMP),
                     Column("updatetime", TIMESTAMP),
                     Column("frozentype", String(20)),
                     Column("devnum", INTEGER),
                     )
