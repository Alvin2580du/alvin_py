from sqlalchemy import TEXT, Table, Column, INT, VARCHAR, TIMESTAMP

from pyduyp.datasources.basic_db import metadata

xiaohua = Table("xiaohua", metadata,
                Column('id', INT, primary_key=True),
                Column("name", VARCHAR),
                Column("namemd5", VARCHAR),

                Column("status", INT),
                Column("createtime", TIMESTAMP)
                )

cailianshe = Table("cailianshe", metadata,
                   Column('id', INT, primary_key=True),
                   Column("news", VARCHAR),
                   Column("createtime", TIMESTAMP),
                   Column("comment", VARCHAR),
                   )
