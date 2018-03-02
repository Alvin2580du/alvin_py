import sqlite3
import pandas as pd

conn = sqlite3.connect("./datasets/Backup.db")
print(dir(conn))
c = conn.cursor()
c.execute("select name from sqlite_master where type = 'table' order by name").fetchall()